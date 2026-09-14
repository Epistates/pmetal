//! Numerical-parity test for the Rust GPT-OSS port against the authoritative
//! HuggingFace `transformers` `GptOssForCausalLM` oracle.
//!
//! Exercises the three forward-pass fixes together — attention **sinks**,
//! **YARN** per-dim RoPE frequencies + embedding mscale, and the **biased
//! router** (plus the clamped `(up + 1)·gate·σ(1.702·gate)` GLU) — end to end.
//! The tiny 2-layer fixture is dumped by
//! `.strategy/parity/dump_gpt_oss_reference.py` and loaded here through the
//! production `load_generic_weights` path.
//!
//! GPT-OSS is the one architecture whose `transformers` weight layout is not
//! pmetal's: experts ship fused, transposed, and with gate/up **interleaved**
//! on the last axis, so the dumper de-interleaves and transposes them into
//! per-expert `nn.Linear` tensors. Getting that wrong is not subtle — it moves
//! the logits by O(1).
//!
//! The synthetic config enables sliding+full attention layers, YARN factor=4,
//! and seeded non-zero per-head sinks so a missing sink term or a wrong RoPE
//! rotation shifts the argmax.

mod common;

use common::{fixture_path, load_shard, ref_tensor};

use pmetal_bridge::compat::Array;
use pmetal_mlx::speculative::SpecCapture;
use pmetal_mlx::test_utils::{
    ParityReport, Tolerance, argmax_last_axis, print_report_table, to_f32_vec_eval,
};
use serial_test::serial;

use pmetal_models::architectures::gpt_oss::{GptOssConfig, GptOssForCausalLM};
use pmetal_models::loader::load_generic_weights;

/// Mirrors `SYNTHETIC_ARGS` in the Python dumper. Fields must stay in sync.
fn synthetic_config_json() -> &'static str {
    r#"{
        "model_type": "gpt_oss",
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "rope_scaling": {
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 32,
            "beta_fast": 32.0,
            "beta_slow": 1.0
        },
        "num_local_experts": 4,
        "experts_per_token": 2,
        "num_experts_per_tok": 2,
        "sliding_window": 4,
        "layer_types": ["sliding_attention", "full_attention"],
        "swiglu_limit": 7.0
    }"#
}

/// Against the definition oracle every checkpoint lands at 1e-7 — within 2x of
/// the oracle's own fp32-vs-fp64 error — so these are ~20x the observed diff
/// rather than the 1000x slack a third-party port needed. `passed()` is
/// `abs || rel`, so the rtols move with the atols.
fn synthetic_tolerances() -> Vec<(&'static str, Tolerance)> {
    vec![
        ("layer_0_hidden", Tolerance::new(1e-5, 1e-5)),
        ("layer_1_hidden", Tolerance::new(1e-5, 1e-5)),
        ("final_hidden", Tolerance::new(1e-5, 1e-5)),
        ("logits", Tolerance::new(1e-5, 1e-5)),
    ]
}

fn compare(
    name: &str,
    rust: &Array,
    reference: &Array,
    tol_table: &[(&str, Tolerance)],
) -> ParityReport {
    let tol = tol_table
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, t)| *t)
        .unwrap_or(Tolerance::new(1e-5, 1e-5));
    if name == "logits" || name == "final_hidden" || name.ends_with("_hidden") {
        ParityReport::compute_with_per_position(name, rust, reference, tol)
    } else {
        ParityReport::compute(name, rust, reference, tol)
    }
}

#[test]
#[serial]
fn gpt_oss_synthetic_parity() {
    let ref_shard = load_shard(&fixture_path("gpt_oss_synth_reference.safetensors"));
    let input_ids = ref_tensor(&ref_shard, "input_ids").clone();

    // Build the model and load the fixture through the production loader
    // (silent-skip `assign_loaded_weights` + eval), reading from a real
    // `model.safetensors` on disk exactly like a checkpoint dir.
    let config: GptOssConfig = json5::from_str(synthetic_config_json()).expect("config parses");
    let mut model = GptOssForCausalLM::new(config).expect("model builds");

    let tmp = tempfile::tempdir().expect("tempdir");
    std::fs::copy(
        fixture_path("gpt_oss_synth_weights.safetensors"),
        tmp.path().join("model.safetensors"),
    )
    .expect("copy weights fixture");
    load_generic_weights(&mut model, tmp.path()).expect("weight loader runs");
    model.init_stacked_moe().expect("stacked MoE init");

    // Production capture path: per-layer hidden taps + post-norm final hidden,
    // then the plain (untied, no-softcap) LM head for logits.
    let mut capture = SpecCapture::with_layers_and_embedding(vec![0, 1], false);
    let final_hidden = model
        .model
        .forward_with_capture(&input_ids, None, None, None, Some(&mut capture))
        .expect("forward_with_capture runs");
    let logits = model.lm_head.forward(&final_hidden);

    let layer0 = capture
        .hidden_states
        .get(&0)
        .expect("layer 0 tap captured")
        .clone();
    let layer1 = capture
        .hidden_states
        .get(&1)
        .expect("layer 1 tap captured")
        .clone();

    let tol = synthetic_tolerances();
    let reports = vec![
        compare(
            "layer_0_hidden",
            &layer0,
            ref_tensor(&ref_shard, "layer_0_hidden"),
            &tol,
        ),
        compare(
            "layer_1_hidden",
            &layer1,
            ref_tensor(&ref_shard, "layer_1_hidden"),
            &tol,
        ),
        compare(
            "final_hidden",
            &final_hidden,
            ref_tensor(&ref_shard, "final_hidden"),
            &tol,
        ),
        compare("logits", &logits, ref_tensor(&ref_shard, "logits"), &tol),
    ];

    println!("\n== GPT-OSS synthetic parity report ==");
    print_report_table(&reports);

    let argmax_rust = argmax_last_axis(&logits);
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

    let failures: Vec<String> = reports
        .iter()
        .filter(|r| !r.passed())
        .map(|r| r.name.clone())
        .collect();
    assert!(
        failures.is_empty(),
        "GPT-OSS synthetic parity failed at checkpoints: {failures:?}"
    );
    assert_eq!(
        argmax_matches,
        argmax_rust.len(),
        "GPT-OSS synthetic parity: argmax mismatch at some positions"
    );
}
