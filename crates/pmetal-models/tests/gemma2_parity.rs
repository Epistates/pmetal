//! Numerical-parity test for the Rust Gemma 2 port.
//!
//! Loads a tiny 2-layer seeded fixture dumped from `mlx_lm.models.gemma2`
//! (`.strategy/parity/dump_gemma2_reference.py`) through the full
//! `DynamicModel::load` dispatcher path — with a `config.json` whose
//! `model_type` is `"gemma2"` but with NO `is_gemma2` flag — so the test
//! validates the dispatcher auto-detection fix end to end: without it the
//! checkpoint would run the Gemma-v1 path (no 4-norm block, no attention
//! softcap, no final-logit softcap) and the argmax would diverge.
//!
//! mlx-lm's gemma2 uses a single global causal mask, so the short synthetic
//! sequence (sliding window is a no-op) is where mlx-lm and pmetal agree.

mod common;

use common::{fixture_path, load_shard, ref_tensor};

use pmetal_mlx::speculative::SpecCapture;
use pmetal_mlx::test_utils::{
    ParityReport, Tolerance, argmax_last_axis, print_report_table, to_f32_vec_eval,
};

use pmetal_models::dispatcher::DynamicModel;

fn synthetic_config_json() -> &'static str {
    r#"{
        "model_type": "gemma2",
        "vocab_size": 512,
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "max_position_embeddings": 8192,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "attn_logit_softcapping": 50.0,
        "final_logit_softcapping": 30.0,
        "query_pre_attn_scalar": 256
    }"#
}

#[test]
fn gemma2_synthetic_parity() {
    let ref_shard = load_shard(&fixture_path("gemma2_synth_reference.safetensors"));
    let input_ids = ref_tensor(&ref_shard, "input_ids").clone();

    // Stage a synthetic model dir (config.json + model.safetensors) and load
    // it through the production dispatcher.
    let tmp = std::env::temp_dir().join(format!("pmetal_gemma2_parity_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).expect("temp dir");
    std::fs::write(tmp.join("config.json"), synthetic_config_json()).expect("write config");
    std::fs::copy(
        fixture_path("gemma2_synth_weights.safetensors"),
        tmp.join("model.safetensors"),
    )
    .expect("copy weights");

    let dyn_model = DynamicModel::load(&tmp).expect("dispatcher loads gemma2");
    let _ = std::fs::remove_dir_all(&tmp);
    let mut model = match dyn_model {
        DynamicModel::Gemma(m) => m,
        other => panic!("expected DynamicModel::Gemma, got {other:?}"),
    };
    assert!(
        model.config().is_gemma2,
        "dispatcher must auto-set is_gemma2 for model_type=gemma2"
    );

    let mut capture = SpecCapture::with_layers_and_embedding(vec![0, 1], true);
    let logits = model
        .forward_with_capture(&input_ids, None, None, &mut capture)
        .expect("forward_with_capture");

    // gemma.rs's forward_with_capture records per-layer hidden states but not
    // an embedding tap; the layer taps + logits + argmax fully constrain the
    // gemma2 path (4-norm block, attn softcap, query_pre_attn scale, embed
    // scale, final-logit softcap).
    let layer0 = capture.hidden_states.get(&0).expect("layer 0 tap").clone();
    let layer1 = capture.hidden_states.get(&1).expect("layer 1 tap").clone();

    let reports = vec![
        ParityReport::compute(
            "layer_0_hidden",
            &layer0,
            ref_tensor(&ref_shard, "layer_0_hidden"),
            Tolerance::new(5e-4, 1e-3),
        ),
        ParityReport::compute_with_per_position(
            "layer_1_hidden",
            &layer1,
            ref_tensor(&ref_shard, "layer_1_hidden"),
            Tolerance::new(1e-3, 2e-3),
        ),
        ParityReport::compute_with_per_position(
            "logits",
            &logits,
            ref_tensor(&ref_shard, "logits"),
            Tolerance::new(5e-3, 5e-3),
        ),
    ];

    println!("\n== Gemma 2 synthetic parity report ==");
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
    assert!(
        failures.is_empty(),
        "Gemma 2 parity failed at: {failures:?}"
    );
    assert_eq!(
        argmax_matches,
        argmax_rust.len(),
        "Gemma 2 parity: argmax mismatch (rust={argmax_rust:?}, ref={argmax_ref:?})"
    );
}
