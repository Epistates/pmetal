//! Numerical-parity test for the Rust Gemma 4 port.
//!
//! Runs two profiles:
//!
//! 1. **Synthetic** (always) — loads a tiny 2-layer seeded fixture checked
//!    into `tests/fixtures/` and compares every tapped checkpoint against
//!    reference tensors from HuggingFace `transformers` 5.10.0.dev0
//!    `Gemma4ForCausalLM` (torch 2.12, seed 42, 12 tokens; a dense and a
//!    routed-experts variant). Catches architectural bugs and weight-loader
//!    bugs under tight tolerances.
//!
//! The 2-layer geometry makes every Gemma-4 asymmetry visible: layer 0 is
//! `sliding_attention` (head_dim 32, full rotary, θ1e4, its own `v_proj`) and
//! layer 1 is `full_attention` (global_head_dim 64, 0.25 partial rotary, θ1e6,
//! `attention_k_eq_v` so there is no `v_proj`). Collapsing the two geometries
//! cannot pass both layers.
//! 2. **Real 31B** (only when `PMETAL_GEMMA4_REFERENCE` is set) — loads a
//!    reference safetensors file dumped from a `gemma-4-31B` checkpoint and
//!    compares it against the Rust forward of the same model. Tolerances
//!    loosened for 60-layer bf16 cumulative drift.
//!
//! Both profiles use the same helper functions from `pmetal_mlx::test_utils`.

mod common;

use common::{fixture_path, load_shard, ref_tensor};

use std::collections::HashMap;
use std::path::PathBuf;

use pmetal_mlx::speculative::SpecCapture;
use pmetal_mlx::test_utils::{
    ParityReport, Tolerance, argmax_last_axis, print_report_table, to_f32_vec_eval,
};
use pmetal_runtime::compat::{Array, ops};

use pmetal_models::architectures::gemma4::{Gemma4Config, Gemma4ForCausalLM, load_gemma4_weights};

/// The synthetic config the fixture was generated with. Changing a field
/// here without regenerating the fixture breaks the comparison.
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
///
/// Against the definition oracle the fp32 fixture lands near the noise floor:
/// embeddings are bit-exact, the sliding layer is 1.2e-7, and logits are 6e-6.
/// `layer_1_hidden` is the outlier at 4e-5 — it is the `full_attention` layer,
/// where the 0.25 partial-rotary split and `attention_k_eq_v` reuse give the
/// two implementations more room to reassociate — so it keeps a looser atol
/// with ~12x headroom while everything else tightens by 100x. The `OR` in
/// `ParityReport::passed()` means the rtols have to move with the atols or
/// they silently keep the old slack.
fn synthetic_tolerances() -> Vec<(&'static str, Tolerance)> {
    vec![
        ("post_embed", Tolerance::new(1e-6, 1e-6)),
        ("layer_0_hidden", Tolerance::new(1e-5, 1e-5)),
        ("layer_1_hidden", Tolerance::new(5e-4, 5e-5)),
        ("final_hidden", Tolerance::new(1e-4, 1e-5)),
        ("logits", Tolerance::new(1e-4, 1e-5)),
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

/// MoE synthetic config the fixture was generated with (same geometry as the
/// dense fixture with the routed-experts branch enabled).
fn moe_synthetic_config_json() -> &'static str {
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
        "enable_moe_block": true,
        "num_experts": 4,
        "top_k_experts": 2,
        "moe_intermediate_size": 64
    }"#
}

/// The E2B/E4B shape: six layers, the last two sharing the KV cache of the
/// last layer of their own attention type (3 for sliding, 2 for full), with
/// per-layer input embeddings. The native engine's only fixture for either.
fn shared_synthetic_config_json() -> &'static str {
    r#"{
        "model_type": "gemma4_text",
        "vocab_size": 512,
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 6,
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
        "layer_types": [
            "sliding_attention", "sliding_attention", "full_attention",
            "sliding_attention", "sliding_attention", "full_attention"
        ],
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
        "hidden_size_per_layer_input": 16,
        "vocab_size_per_layer_input": 512,
        "num_kv_shared_layers": 2,
        "use_double_wide_mlp": false,
        "enable_moe_block": false
    }"#
}

/// Build the Rust Gemma4 model from a config JSON + weight fixture and return
/// it ready for forward passes.
fn build_model(config_json: &str, weights_fixture: &str) -> Gemma4ForCausalLM {
    let config: Gemma4Config = json5::from_str(config_json).expect("config parses");
    let mut model = Gemma4ForCausalLM::new(config).expect("model builds");
    let weights = load_shard(&fixture_path(weights_fixture));
    let report = load_gemma4_weights(&mut model, &weights).expect("weight loader runs");
    assert!(
        report.loaded > 0,
        "weight loader loaded 0 tensors (skipped={:?})",
        report.skipped
    );
    model
}

/// Read `input_ids` from a reference shard. They are stored as int32
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
    let wants_per_position = (name.starts_with("layer_") && name.ends_with("_hidden"))
        || name == "logits"
        || name == "final_hidden";
    if wants_per_position {
        ParityReport::compute_with_per_position(name, rust, reference, tol)
    } else {
        ParityReport::compute(name, rust, reference, tol)
    }
}

#[test]
fn gemma4_synthetic_parity() {
    run_synthetic_parity(
        "dense",
        synthetic_config_json(),
        "gemma4_synth_weights.safetensors",
        "gemma4_synth_reference.safetensors",
    );
}

/// MoE-block parity: exercises the Phase A routed-experts port (parallel dense
/// MLP + top-k experts) against the `transformers` `Gemma4TextExperts` oracle.
/// Same tight tolerances as the dense fixture — the 2-layer config has minimal
/// drift, and transformers' expert layout (`gate_up_proj [E, 2·I, H]` +
/// `down_proj [E, H, I]`) is already pmetal's fused layout, so the fixture
/// needs no weight remapping.
#[test]
fn gemma4_moe_synthetic_parity() {
    run_synthetic_parity(
        "MoE",
        moe_synthetic_config_json(),
        "gemma4_moe_synth_weights.safetensors",
        "gemma4_moe_synth_reference.safetensors",
    );
}

/// The native Gemma 4 engine against the same `transformers` oracle, on the
/// dense fixture. The native engine is a separate implementation from the
/// module-based port above, so it needs its own check.
#[test]
fn gemma4_native_synthetic_parity() {
    run_native_synthetic_parity(
        "dense",
        synthetic_config_json(),
        "gemma4_synth_weights.safetensors",
        "gemma4_synth_reference.safetensors",
    );
}

/// The native engine's KV-sharing layers and per-layer inputs (E2B/E4B)
/// against `transformers`, over 40 tokens so the window of 8 is long past.
#[test]
fn gemma4_native_shared_kv_synthetic_parity() {
    run_native_synthetic_parity(
        "shared-KV",
        shared_synthetic_config_json(),
        "gemma4_shared_synth_weights.safetensors",
        "gemma4_shared_synth_reference.safetensors",
    );
}

#[test]
fn gemma4_native_shared_kv_chunked_prefill_matches_transformers() {
    run_native_chunked_prefill(
        "shared-KV",
        shared_synthetic_config_json(),
        "gemma4_shared_synth_weights.safetensors",
        "gemma4_shared_synth_reference.safetensors",
    );
}

/// The native engine's routed-experts branch against `transformers`
/// `Gemma4TextRouter` and `Gemma4TextExperts`, through the fused
/// `gate_up_proj` layout Hugging Face checkpoints ship.
#[test]
fn gemma4_native_moe_synthetic_parity() {
    run_native_synthetic_parity(
        "MoE",
        moe_synthetic_config_json(),
        "gemma4_moe_synth_weights.safetensors",
        "gemma4_moe_synth_reference.safetensors",
    );
}

/// The fixture as a checkpoint directory, loaded by the native engine. The
/// directory is returned so it outlives the weights' lazily read tensors.
fn native_checkpoint(
    config_json: &str,
    weights_fixture: &str,
) -> (
    tempfile::TempDir,
    pmetal_runtime::gemma4_native::Gemma4Config,
    pmetal_runtime::gemma4_native::NativeWeights,
) {
    use pmetal_runtime::gemma4_native;

    // The native loader reads a checkpoint directory, not a tensor map.
    let dir = tempfile::tempdir().expect("temp dir");
    std::fs::write(dir.path().join("config.json"), config_json).expect("write config");
    std::fs::copy(
        fixture_path(weights_fixture),
        dir.path().join("model.safetensors"),
    )
    .expect("stage weights");

    let config = gemma4_native::load_config(dir.path()).expect("native config parses");
    let weights = gemma4_native::load_model(dir.path(), &config).expect("native load");
    (dir, config, weights)
}

/// A prompt in prefill passes of every size around the sliding window (8) and
/// the prompt length (12): the last position's logits, and the cache the
/// passes leave, must be the whole-prompt forward's, which is `transformers`'.
#[test]
fn gemma4_native_chunked_prefill_matches_transformers() {
    run_native_chunked_prefill(
        "dense",
        synthetic_config_json(),
        "gemma4_synth_weights.safetensors",
        "gemma4_synth_reference.safetensors",
    );
}

#[test]
fn gemma4_native_moe_chunked_prefill_matches_transformers() {
    run_native_chunked_prefill(
        "MoE",
        moe_synthetic_config_json(),
        "gemma4_moe_synth_weights.safetensors",
        "gemma4_moe_synth_reference.safetensors",
    );
}

/// A cached decode far past the sliding window (8) against `transformers`,
/// one token at a time after a prefill in passes: the sliding layers keep a
/// window and a little more, move their buffer forward as the decode goes on
/// (a slack of 1, 4 and the default, so the buffer moves every token, every
/// few, and not at all), and every step must still be the reference's row.
/// Covers KV-sharing layers reading those buffers too.
#[test]
fn gemma4_native_cached_decode_past_the_window_matches_transformers() {
    use pmetal_runtime::decode::prefill_last_logits;
    use pmetal_runtime::gemma4_native::{self, forward_step_last};

    let (_dir, config, weights) = native_checkpoint(
        shared_synthetic_config_json(),
        "gemma4_shared_synth_weights.safetensors",
    );
    let ref_shard = load_shard(&fixture_path("gemma4_shared_synth_reference.safetensors"));
    let ids: Vec<u32> = to_f32_vec_eval(&input_ids_from_shard(&ref_shard))
        .into_iter()
        .map(|v| v as u32)
        .collect();
    let want = ref_tensor(&ref_shard, "logits");
    let tolerance = Tolerance::new(1.5e-5, 0.0);
    let window = config.sliding_window;

    for (prefill, slack, detach) in [
        (10usize, 1, true),
        (10, 4, true),
        (10, gemma4_native::SLIDING_SLACK, true),
        (10, 4, false),
        (3, 1, false),
    ] {
        let mut cache = gemma4_native::build_cache(&weights, &config);
        for layer in &mut cache.layers {
            layer.slack = slack;
        }
        prefill_last_logits(&weights, &mut cache, &ids[..prefill], 4, forward_step_last).eval();
        if detach {
            cache.eval_and_detach_states();
            cache.reserve_decode_inputs((ids.len() - prefill) as i32, weights.model_dtype);
        }
        let mut reports = Vec::new();
        for pos in prefill..ids.len() {
            let step = forward_step_last(
                &weights,
                &pmetal_runtime::decode::prompt_tokens_to_input(&ids[pos..pos + 1]),
                &mut cache,
            );
            pmetal_runtime::check_unobserved_error().expect("native decode step");
            let row = pmetal_runtime::compat::ops::slice_axis(want, 1, pos as i32, pos as i32 + 1);
            let name = format!("prefill {prefill}, slack {slack}, step {pos}");
            reports.push(compare_checkpoint(
                &name,
                &step.reshape(&[1, 1, -1]),
                &row,
                &[(&name, tolerance)],
            ));
        }
        print_report_table(&reports);
        assert!(
            reports.iter().all(|r| r.passed()),
            "prefill {prefill}, slack {slack}, detach {detach}: decode out of tolerance"
        );

        // The sliding layers hold a window and the slack, however long the
        // decode ran; the full layers hold everything.
        for (i, layer) in cache.layers.iter().enumerate() {
            let Some(keys) = layer.keys.as_ref() else {
                continue; // a KV-sharing layer reads another's
            };
            match layer.window {
                Some(_) => assert!(
                    keys.dim(2) <= window + slack.max(1) + 4,
                    "layer {i}: a sliding layer holds {} keys after {} tokens",
                    keys.dim(2),
                    layer.offset
                ),
                None => assert!(keys.dim(2) >= layer.offset as i32),
            }
        }
    }
}

/// A bf16 checkpoint stays bf16 through a multi-token pass: the residual
/// stream, the logits and every layer's cache. A pass that promoted one op to
/// f32 carries f32 through every layer after it (each matmul then casts its
/// bf16 weights per op), writes f32 keys and values if the cache takes the
/// pass's dtype, and returns f32 logits.
#[test]
fn gemma4_native_prefill_stays_in_the_models_dtype() {
    use pmetal_runtime::compat::Dtype;
    use pmetal_runtime::gemma4_native;

    let bf16 = Dtype::Bfloat16.as_i32();
    for (label, config_json, weights_fixture, reference) in [
        (
            "dense",
            synthetic_config_json(),
            "gemma4_synth_weights.safetensors",
            "gemma4_synth_reference.safetensors",
        ),
        (
            "per-layer inputs and shared KV",
            shared_synthetic_config_json(),
            "gemma4_shared_synth_weights.safetensors",
            "gemma4_shared_synth_reference.safetensors",
        ),
        (
            "MoE",
            moe_synthetic_config_json(),
            "gemma4_moe_synth_weights.safetensors",
            "gemma4_moe_synth_reference.safetensors",
        ),
    ] {
        // The fixture's weights, cast to bf16.
        let dir = tempfile::tempdir().expect("temp dir");
        std::fs::write(dir.path().join("config.json"), config_json).expect("write config");
        let tensors: Vec<(String, Array)> = load_shard(&fixture_path(weights_fixture))
            .into_iter()
            .map(|(k, v)| {
                let cast = if v.ndim() > 0 { v.as_dtype(bf16) } else { v };
                (k, cast)
            })
            .collect();
        let entries: Vec<(&str, &Array)> = tensors.iter().map(|(k, v)| (k.as_str(), v)).collect();
        Array::save_safetensors(
            dir.path().join("model.safetensors").to_str().unwrap(),
            &entries,
        )
        .expect("save the weights");
        let config = gemma4_native::load_config(dir.path()).expect("config parses");
        let weights = gemma4_native::load_model(dir.path(), &config).expect("native load");
        assert_eq!(weights.model_dtype, bf16, "{label}: the checkpoint is bf16");

        let ids = input_ids_from_shard(&load_shard(&fixture_path(reference)));
        let mut cache = gemma4_native::build_cache(&weights, &config);
        let logits = gemma4_native::forward_step(&weights, &ids, &mut cache);
        pmetal_runtime::check_unobserved_error().expect("native forward");
        assert_eq!(logits.dtype_raw(), bf16, "{label}: logits dtype");
        for (i, layer) in cache.layers.iter().enumerate() {
            for (name, array) in [("keys", &layer.keys), ("values", &layer.values)] {
                if let Some(array) = array {
                    assert_eq!(array.dtype_raw(), bf16, "{label}: layer {i} {name}");
                }
            }
        }
    }
}

/// A long prompt's prefill holds the memory of a pass, not of the prompt:
/// the engine's own prefill, against a whole-prompt forward that computes
/// every position's logits as `forward_step` returns them.
#[test]
#[serial_test::serial]
fn gemma4_native_prefill_memory_is_bounded_by_the_pass() {
    use pmetal_runtime::decode::{SamplingParams, prefill_first_token};
    use pmetal_runtime::gemma4_native::{self, forward_step};
    use pmetal_runtime::inline_array::{clear_cache, get_peak_memory, reset_peak_memory};

    let (_dir, config, weights) =
        native_checkpoint(synthetic_config_json(), "gemma4_synth_weights.safetensors");
    // 16384 tokens, 8 passes: its logits alone are 16384 x 512 x 4 bytes.
    let prompt: Vec<u32> = (0..16384).map(|i| (i * 7 % 512) as u32).collect();
    let peak = |prefill: &dyn Fn(&mut gemma4_native::NativeCache) -> u32| {
        let mut cache = gemma4_native::build_cache(&weights, &config);
        clear_cache();
        reset_peak_memory();
        let token = prefill(&mut cache);
        pmetal_runtime::check_unobserved_error().expect("native prefill");
        (get_peak_memory(), token)
    };
    let (whole, whole_token) = peak(&|cache| {
        prefill_first_token(
            &weights,
            cache,
            &prompt,
            SamplingParams::new(0.0),
            forward_step,
        )
        .0
    });
    let (chunked, chunked_token) = peak(&|cache| {
        gemma4_native::prefill_first_token(&weights, cache, &prompt, SamplingParams::new(0.0)).0
    });
    println!("prefill peak: whole prompt {whole} bytes, in passes {chunked} bytes");
    assert_eq!(chunked_token, whole_token, "the same first token");
    assert!(
        chunked * 2 < whole,
        "prefill in passes peaked at {chunked} bytes against {whole} for the whole prompt"
    );
}

fn run_native_chunked_prefill(
    label: &str,
    config_json: &str,
    weights_fixture: &str,
    reference_fixture: &str,
) {
    use pmetal_runtime::decode::prefill_last_logits;
    use pmetal_runtime::gemma4_native::{self, forward_step_last};

    let (_dir, config, weights) = native_checkpoint(config_json, weights_fixture);
    let ref_shard = load_shard(&fixture_path(reference_fixture));
    let ids: Vec<u32> = to_f32_vec_eval(&input_ids_from_shard(&ref_shard))
        .into_iter()
        .map(|v| v as u32)
        .collect();
    let want = ref_tensor(&ref_shard, "logits");
    let len = ids.len() as i32;
    let last = pmetal_runtime::compat::ops::slice_axis(want, 1, len - 1, len);
    // The same bound as `run_native_synthetic_parity`'s logits.
    let tolerance = Tolerance::new(1.5e-5, 0.0);

    for chunk in [1, 3, 5, 8, 11, 12, 100] {
        let mut cache = gemma4_native::build_cache(&weights, &config);
        let logits = prefill_last_logits(&weights, &mut cache, &ids, chunk, forward_step_last);
        pmetal_runtime::check_unobserved_error().expect("native chunked prefill");
        let report = compare_checkpoint(
            &format!("{label} chunk {chunk}"),
            &logits.reshape(&[1, 1, -1]),
            &last,
            &[(&format!("{label} chunk {chunk}"), tolerance)],
        );
        print_report_table(std::slice::from_ref(&report));
        assert!(report.passed(), "{label}: chunk {chunk} last logits");

        // The cache the passes leave decodes on: all but the last token in
        // chunks, then the last one as a decode step.
        let mut cache = gemma4_native::build_cache(&weights, &config);
        let head = &ids[..ids.len() - 1];
        prefill_last_logits(&weights, &mut cache, head, chunk, forward_step_last).eval();
        let step = gemma4_native::forward_step_last(
            &weights,
            &pmetal_runtime::decode::prompt_tokens_to_input(&ids[ids.len() - 1..]),
            &mut cache,
        );
        pmetal_runtime::check_unobserved_error().expect("native decode after chunked prefill");
        let report = compare_checkpoint(
            &format!("{label} decode after chunk {chunk}"),
            &step.reshape(&[1, 1, -1]),
            &last,
            &[(&format!("{label} decode after chunk {chunk}"), tolerance)],
        );
        print_report_table(std::slice::from_ref(&report));
        assert!(report.passed(), "{label}: decode after chunk {chunk}");
    }
}

fn run_native_synthetic_parity(
    label: &str,
    config_json: &str,
    weights_fixture: &str,
    reference_fixture: &str,
) {
    use pmetal_runtime::gemma4_native;

    let (_dir, config, weights) = native_checkpoint(config_json, weights_fixture);
    let mut cache = gemma4_native::build_cache(&weights, &config);

    let ref_shard = load_shard(&fixture_path(reference_fixture));
    let logits =
        gemma4_native::forward_step(&weights, &input_ids_from_shard(&ref_shard), &mut cache);
    pmetal_runtime::check_unobserved_error().expect("native forward");

    // ⚠️ Tighter than `synthetic_tolerances`, which the module port can
    // afford because it also checks every layer's hidden state. With only the
    // logits to go on, 1e-4 let routing to the *least* likely experts through
    // (3.3e-5). Measured: 5.4e-6 correct on both fixtures, 3.3e-5 with the
    // top-k inverted, 5.5e-4 without `per_expert_scale`, 1.5e-3 without the
    // dense branch's post-norm. A gate/up swap is invisible here, since the
    // fixture keeps GELU linear; `fused_experts_split_gate_first` pins it.
    let report = compare_checkpoint(
        "logits",
        &logits,
        ref_tensor(&ref_shard, "logits"),
        &[("logits", Tolerance::new(1.5e-5, 0.0))],
    );
    println!("\n== Gemma 4 native {label} synthetic parity report ==");
    print_report_table(std::slice::from_ref(&report));

    let argmax = argmax_last_axis(&logits);
    let argmax_ref: Vec<i32> = to_f32_vec_eval(ref_tensor(&ref_shard, "argmax_tokens"))
        .into_iter()
        .map(|v| v as i32)
        .collect();
    assert!(
        report.passed(),
        "Gemma 4 native {label}: logits outside tolerance"
    );
    assert_eq!(
        argmax, argmax_ref,
        "Gemma 4 native {label}: argmax mismatch"
    );
}

fn run_synthetic_parity(
    label: &str,
    config_json: &str,
    weights_fixture: &str,
    reference_fixture: &str,
) {
    let ref_shard = load_shard(&fixture_path(reference_fixture));
    let input_ids = input_ids_from_shard(&ref_shard);

    let mut model = build_model(config_json, weights_fixture);

    // Use the production `forward_with_capture` path. That goes through
    // `Gemma4Attention::forward` with the internal mask-building logic and
    // whatever attention backend `fused_sdpa` benchmark-selects. Using this
    // path (rather than a manual layer walk) ensures the parity test
    // exercises the same code production code takes.
    let mut capture = SpecCapture::with_layers_and_embedding(vec![0, 1], true);
    let final_hidden_rust = model
        .model
        .forward_with_capture(&input_ids, None, None, None, Some(&mut capture))
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
    let reports = vec![
        compare_checkpoint(
            "post_embed",
            &post_embed_rust,
            ref_tensor(&ref_shard, "post_embed"),
            &tol,
        ),
        compare_checkpoint(
            "layer_0_hidden",
            &layer0_rust,
            ref_tensor(&ref_shard, "layer_0_hidden"),
            &tol,
        ),
        compare_checkpoint(
            "layer_1_hidden",
            &layer1_rust,
            ref_tensor(&ref_shard, "layer_1_hidden"),
            &tol,
        ),
        compare_checkpoint(
            "final_hidden",
            &final_hidden_rust,
            ref_tensor(&ref_shard, "final_hidden"),
            &tol,
        ),
        compare_checkpoint(
            "logits",
            &logits_rust,
            ref_tensor(&ref_shard, "logits"),
            &tol,
        ),
    ];

    println!("\n== Gemma 4 {label} synthetic parity report ==");
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
        "Gemma 4 {label} synthetic parity failed at checkpoints: {failures:?}"
    );
    assert_eq!(
        argmax_matches,
        argmax_rust.len(),
        "Gemma 4 {label} synthetic parity: argmax mismatch at some positions"
    );
}

/// Real 31B parity run, gated behind `PMETAL_GEMMA4_REFERENCE`.
///
/// Set the env var to a reference safetensors file of transformers'
/// `Gemma4ForCausalLM` on a `gemma-4-31B` checkpoint (fp32, eager
/// attention): `input_ids` (int32 `[1, T]`), `post_embed`, `layer_{i}_hidden`
/// for each tapped layer, `final_hidden`, `logits` and `argmax_tokens`. The
/// sidecar `<file>.meta.json` next to it gives `model_id` (a local checkpoint
/// directory, used to load the real weights) and `tap_layers`.
#[test]
fn gemma4_31b_parity() {
    let ref_path = match std::env::var("PMETAL_GEMMA4_REFERENCE") {
        Ok(v) => PathBuf::from(v),
        Err(_) => {
            eprintln!(
                "PMETAL_GEMMA4_REFERENCE not set — skipping gemma4_31b_parity. \
                 To run: set it to transformers taps of a gemma-4-31B \
                 checkpoint (layout in the test's docs)."
            );
            return;
        }
    };
    if !ref_path.exists() {
        panic!("PMETAL_GEMMA4_REFERENCE points at {ref_path:?} but the file does not exist");
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
             Set the sidecar's `model_id` to an absolute path to a local \
             clone. Skipping."
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
        .forward_with_capture(&input_ids, None, None, None, Some(&mut capture))
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
