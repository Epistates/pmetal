//! Numerical-parity test for the Rust DiffusionGemma **encoder** port.
//!
//! Unlike the Gemma 4 parity test (which uses mlx-lm as the oracle),
//! DiffusionGemma has no mlx-lm implementation, so the reference tensors are
//! dumped from **transformers nightly** (`transformers >= 5.8.0.dev0`) by
//! `.strategy/parity/dump_diffusion_gemma_reference.py`. That script builds a
//! tiny seeded `DiffusionGemmaEncoderTextModel`, captures the scaled
//! embeddings, every encoder-layer output, and the final-normed hidden state,
//! and writes both those tensors and the raw HF weights into
//! `tests/fixtures/`.
//!
//! This test loads the same weights into the Rust
//! [`DiffusionGemmaEncoderModel`] and diffs each checkpoint. It exercises the
//! full 7-norm parallel dense+MoE layer, the fp32 router + grouped experts,
//! and the sliding/full dual attention geometry (reused from the
//! parity-verified Gemma 4 blocks).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use pmetal_bridge::compat::Array;
use pmetal_mlx::test_utils::{ParityReport, Tolerance, print_report_table};

use pmetal_models::architectures::diffusion_gemma::{
    DiffusionGemmaEncoderModel, DiffusionGemmaTextConfig, load_diffusion_gemma_encoder_weights,
};

fn load_shard(path: &Path) -> HashMap<String, Array> {
    let path_str = path.to_str().expect("utf8 path");
    let pairs = pmetal_bridge::inline_array::load_safetensors_shard(path_str)
        .unwrap_or_else(|| panic!("failed to load safetensors shard at {path_str:?}"));
    pairs.into_iter().collect()
}

fn ref_tensor<'a>(shard: &'a HashMap<String, Array>, key: &str) -> &'a Array {
    shard
        .get(key)
        .unwrap_or_else(|| panic!("reference shard missing key {key:?}"))
}

fn fixture_path(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push(name);
    p
}

/// Synthetic config — mirrors `SYNTHETIC_ARGS` in the Python dumper. The RoPE
/// parameters are left `None` so [`DiffusionGemmaTextConfig::resolved_rope_parameters`]
/// supplies the same defaults the dumper passes explicitly (full: 0.25 / θ1e6 /
/// proportional; sliding: 1.0 / θ1e4 / default).
fn synthetic_config() -> DiffusionGemmaTextConfig {
    DiffusionGemmaTextConfig {
        vocab_size: 512,
        hidden_size: 128,
        intermediate_size: 256,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 32,
        global_head_dim: 64,
        num_global_key_value_heads: Some(1),
        sliding_window: 8,
        sliding_window_pattern: 2,
        layer_types: vec![
            "sliding_attention".to_string(),
            "full_attention".to_string(),
        ],
        rms_norm_eps: 1e-6,
        final_logit_softcapping: Some(30.0),
        num_experts: 8,
        top_k_experts: 2,
        moe_intermediate_size: 32,
        ..Default::default()
    }
}

fn synthetic_tolerances() -> Vec<(&'static str, Tolerance)> {
    vec![
        ("post_embed", Tolerance::new(1e-4, 1e-4)),
        ("layer_0_hidden", Tolerance::new(2e-3, 2e-3)),
        // Full-attention layer: proportional RoPE + MoE accumulate the most
        // cross-backend (torch vs MLX) fp32 drift; observed ~3.8e-3.
        ("layer_1_hidden", Tolerance::new(6e-3, 3e-3)),
        ("final_hidden", Tolerance::new(4e-3, 3e-3)),
    ]
}

fn compare(name: &str, rust: &Array, reference: &Array, tol: &[(&str, Tolerance)]) -> ParityReport {
    let t = tol
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, t)| *t)
        .unwrap_or(Tolerance::new(1e-3, 1e-3));
    ParityReport::compute_with_per_position(name, rust, reference, t)
}

#[test]
fn diffusion_gemma_encoder_synthetic_parity() {
    let ref_shard = load_shard(&fixture_path(
        "diffusion_gemma_encoder_reference.safetensors",
    ));
    let weights = load_shard(&fixture_path("diffusion_gemma_encoder_weights.safetensors"));
    let input_ids = ref_tensor(&ref_shard, "input_ids").clone();

    let config = synthetic_config();
    let mut model = DiffusionGemmaEncoderModel::new(config.clone()).expect("model builds");
    let report = load_diffusion_gemma_encoder_weights(&mut model, &weights)
        .expect("encoder weight loader runs");
    assert!(
        report.skipped.is_empty(),
        "encoder weight loader skipped tensors: {:?}",
        report.skipped
    );
    assert!(report.loaded > 0, "encoder weight loader loaded 0 tensors");

    // Manually walk the encoder via the same public `forward_encoder` method
    // the production `forward` uses, capturing each layer's output. This is the
    // production code path — `DiffusionGemmaEncoderModel::forward` is a thin
    // loop over exactly these calls.
    let scale = Array::from_f32((config.hidden_size as f32).sqrt());
    let mut h = model.embed_tokens.forward(&input_ids).multiply(&scale);
    let post_embed_rust = h.clone();

    let mut layer_hidden: Vec<Array> = Vec::new();
    for layer in model.layers.iter_mut() {
        let (next, _k, _v) = layer
            .forward_encoder(&h, None, 0)
            .expect("encoder layer forward runs");
        layer_hidden.push(next.clone());
        h = next;
    }
    let final_hidden_rust = model.norm.forward(&h);

    let tol = synthetic_tolerances();
    let mut reports = vec![compare(
        "post_embed",
        &post_embed_rust,
        ref_tensor(&ref_shard, "post_embed"),
        &tol,
    )];
    for (i, hidden) in layer_hidden.iter().enumerate() {
        let key = format!("layer_{i}_hidden");
        reports.push(compare(&key, hidden, ref_tensor(&ref_shard, &key), &tol));
    }
    reports.push(compare(
        "final_hidden",
        &final_hidden_rust,
        ref_tensor(&ref_shard, "final_hidden"),
        &tol,
    ));

    println!("\n== DiffusionGemma encoder synthetic parity report ==");
    print_report_table(&reports);

    // Cross-check: the production `forward` must agree with the manual walk's
    // final hidden state (guards against the loop drifting from `forward`).
    let mut prod_model = DiffusionGemmaEncoderModel::new(config.clone()).expect("model builds");
    load_diffusion_gemma_encoder_weights(&mut prod_model, &weights).expect("weights load");
    let (prod_final, _kvs) = prod_model
        .forward(&input_ids)
        .expect("production forward runs");
    let prod_report = ParityReport::compute(
        "forward_vs_walk",
        &prod_final,
        &final_hidden_rust,
        Tolerance::new(1e-6, 1e-6),
    );
    assert!(
        prod_report.passed(),
        "production forward() diverges from manual layer walk (max abs {:.3e})",
        prod_report.max_abs_diff
    );

    let failures: Vec<String> = reports
        .iter()
        .filter(|r| !r.passed())
        .map(|r| r.name.clone())
        .collect();
    assert!(
        failures.is_empty(),
        "DiffusionGemma encoder parity failed at checkpoints: {failures:?}"
    );
}
