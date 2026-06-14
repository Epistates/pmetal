//! Numerical-parity test for the Rust DiffusionGemma **decoder** port.
//!
//! The decoder refines a `canvas_length` block with bidirectional attention
//! while reading the encoder's read-only K/V cache. To isolate the decoder
//! forward from any encoder drift, the reference dumper
//! (`.strategy/parity/dump_diffusion_gemma_reference.py --mode decoder`) runs
//! the transformers-nightly oracle, captures the encoder's per-layer KV cache
//! as fixtures, and feeds it straight into the oracle decoder. This test loads
//! the *same* cache tensors into the Rust [`DiffusionGemmaDecoderModel`], so a
//! mismatch is purely decoder logic — self-conditioning, the bidirectional
//! `[encoder_kv | canvas]` attention, and the shared 7-norm MoE layer.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use pmetal_bridge::compat::{Array, ops};
use pmetal_mlx::test_utils::{ParityReport, Tolerance, print_report_table};

use pmetal_models::architectures::diffusion_gemma::{
    DiffusionGemmaDecoderModel, DiffusionGemmaTextConfig, load_diffusion_gemma_decoder_weights,
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

/// Synthetic config — mirrors `SYNTHETIC_ARGS` in the Python dumper (same as
/// the encoder test). The forward derives canvas/enc lengths from the dumped
/// tensors, so `canvas_length` here is irrelevant.
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

fn tolerances() -> Vec<(&'static str, Tolerance)> {
    vec![
        ("post_self_cond", Tolerance::new(1e-3, 1e-3)),
        ("post_self_cond_zeros", Tolerance::new(1e-3, 1e-3)),
        ("layer_0_hidden", Tolerance::new(3e-3, 3e-3)),
        // Full bidirectional attention over [enc|canvas] + MoE: most drift.
        ("layer_1_hidden", Tolerance::new(6e-3, 3e-3)),
        ("final_hidden", Tolerance::new(5e-3, 3e-3)),
        ("logits", Tolerance::new(6e-3, 3e-3)),
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

fn build_model(weights: &HashMap<String, Array>) -> DiffusionGemmaDecoderModel {
    let mut model = DiffusionGemmaDecoderModel::new(synthetic_config()).expect("model builds");
    let report =
        load_diffusion_gemma_decoder_weights(&mut model, weights).expect("decoder weight loader");
    assert!(
        report.skipped.is_empty(),
        "decoder weight loader skipped tensors: {:?}",
        report.skipped
    );
    assert!(report.loaded > 0, "decoder weight loader loaded 0 tensors");
    model
}

/// Reconstruct the encoder KV cache (`Vec<(keys, values)>`) from the dumped
/// per-layer tensors.
fn encoder_kvs(shard: &HashMap<String, Array>, num_layers: usize) -> Vec<(Array, Array)> {
    (0..num_layers)
        .map(|i| {
            (
                ref_tensor(shard, &format!("cache_layer{i}_keys")).clone(),
                ref_tensor(shard, &format!("cache_layer{i}_values")).clone(),
            )
        })
        .collect()
}

#[test]
fn diffusion_gemma_decoder_synthetic_parity() {
    let ref_shard = load_shard(&fixture_path(
        "diffusion_gemma_decoder_reference.safetensors",
    ));
    let weights = load_shard(&fixture_path("diffusion_gemma_decoder_weights.safetensors"));
    let config = synthetic_config();
    let num_layers = config.num_hidden_layers as usize;

    let decoder_input_ids = ref_tensor(&ref_shard, "decoder_input_ids").clone();
    let sc_logits = ref_tensor(&ref_shard, "self_conditioning_logits").clone();
    let kvs = encoder_kvs(&ref_shard, num_layers);

    let mut model = build_model(&weights);
    let scale = Array::from_f32((config.hidden_size as f32).sqrt());
    let tol = tolerances();
    let mut reports = Vec::new();

    // --- Step >0 path: non-zero self-conditioning signal ---
    // Manual-walk mirroring `forward` to capture taps.
    let inputs_embeds = model
        .embed_tokens
        .forward(&decoder_input_ids)
        .multiply(&scale);
    let probs = ops::softmax_axis(&sc_logits.as_type::<f32>(), -1);
    let weight = model.embed_tokens.weight.as_ref();
    let probs = probs.as_dtype(weight.dtype().as_i32());
    let soft = ops::matmul(&probs, weight).multiply(&scale);
    let mut h = model
        .self_conditioning
        .forward(&inputs_embeds, &soft)
        .expect("self-conditioning runs");
    reports.push(compare(
        "post_self_cond",
        &h,
        ref_tensor(&ref_shard, "post_self_cond"),
        &tol,
    ));

    let canvas = decoder_input_ids.dim(1);
    let canvas_offset = kvs.iter().map(|(k, _)| k.dim(2)).max().unwrap();
    for (i, (enc_k, enc_v)) in kvs.iter().enumerate() {
        let enc_len = enc_k.dim(2);
        let mask = ops::zeros_dtype(&[1, 1, canvas, enc_len + canvas], h.dtype());
        h = model.layers[i]
            .forward_decoder(&h, enc_k, enc_v, Some(&mask), canvas_offset)
            .expect("decoder layer forward");
        reports.push(compare(
            &format!("layer_{i}_hidden"),
            &h,
            ref_tensor(&ref_shard, &format!("layer_{i}_hidden")),
            &tol,
        ));
    }
    let final_hidden_rust = model.norm.forward(&h);
    reports.push(compare(
        "final_hidden",
        &final_hidden_rust,
        ref_tensor(&ref_shard, "final_hidden"),
        &tol,
    ));

    // LM head: tied to the decoder embedding, then fp32 final-logit softcap.
    let raw_logits = model
        .embed_tokens
        .as_linear(&final_hidden_rust)
        .as_type::<f32>();
    let cap = Array::from_f32(config.final_logit_softcapping.unwrap());
    let logits_rust = ops::tanh(&raw_logits.divide(&cap)).multiply(&cap);
    reports.push(compare(
        "logits",
        &logits_rust,
        ref_tensor(&ref_shard, "logits"),
        &tol,
    ));

    // --- Step 0 path: None self-conditioning (zeroed signal) ---
    let zeros = ops::zeros_like(&inputs_embeds);
    let sc_zeros = model
        .self_conditioning
        .forward(&inputs_embeds, &zeros)
        .expect("step-0 self-conditioning runs");
    reports.push(compare(
        "post_self_cond_zeros",
        &sc_zeros,
        ref_tensor(&ref_shard, "post_self_cond_zeros"),
        &tol,
    ));

    println!("\n== DiffusionGemma decoder synthetic parity report ==");
    print_report_table(&reports);

    // Cross-check: production `forward` (with sc logits) agrees with the walk.
    let mut prod = build_model(&weights);
    let prod_final = prod
        .forward(&decoder_input_ids, &kvs, Some(&sc_logits))
        .expect("production decoder forward runs");
    let prod_report = ParityReport::compute(
        "forward_vs_walk",
        &prod_final,
        &final_hidden_rust,
        Tolerance::new(1e-6, 1e-6),
    );
    assert!(
        prod_report.passed(),
        "production forward() diverges from manual walk (max abs {:.3e})",
        prod_report.max_abs_diff
    );

    let failures: Vec<String> = reports
        .iter()
        .filter(|r| !r.passed())
        .map(|r| r.name.clone())
        .collect();
    assert!(
        failures.is_empty(),
        "DiffusionGemma decoder parity failed at checkpoints: {failures:?}"
    );
}
