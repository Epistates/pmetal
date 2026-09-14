//! A cached decode step has to agree with recomputing the whole prefix.
//!
//! Generation runs one token at a time against a KV cache. Everything else —
//! training, parity fixtures, the synthetic oracles — runs the trunk in one
//! uncached pass. So a cache bug is invisible to every other test in the tree
//! and shows up only as a model that produces a sensible first token and then
//! degenerates, which reads like a sampling problem rather than a forward one.
//!
//! Each case decodes greedily for a few steps with a cache and, after every
//! step, recomputes the logits for the same prefix with no cache at all. The
//! two have to agree at the last position. Tiny random-init models: the
//! invariant is about the cache, not the weights.
//!
//! Everything goes through `DynamicModel`, which is the path generation
//! actually takes, so a per-architecture cache built with the wrong geometry
//! is in scope here rather than hidden behind a hand-built one.

use pmetal_bridge::compat::Array;
use pmetal_models::dispatcher::DynamicModel;

const PROMPT: [i32; 5] = [3, 11, 7, 2, 9];
const STEPS: usize = 4;

/// One config per architecture family, small enough to build in milliseconds.
/// Sliding windows are wide enough that they never engage over this prefix, so
/// a disagreement is the cache rather than a legitimate mask difference.
const CASES: &[(&str, &str)] = &[
    (
        "llama",
        r#"{
            "model_type": "llama",
            "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 2, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 8,
            "max_position_embeddings": 256, "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0, "tie_word_embeddings": false
        }"#,
    ),
    (
        "qwen2",
        r#"{
            "model_type": "qwen2",
            "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 2, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 8,
            "max_position_embeddings": 256, "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0, "tie_word_embeddings": false
        }"#,
    ),
    (
        "qwen3",
        r#"{
            "model_type": "qwen3",
            "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 2, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 8,
            "max_position_embeddings": 256, "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0, "tie_word_embeddings": false
        }"#,
    ),
    (
        "mistral",
        r#"{
            "model_type": "mistral",
            "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 2, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 8,
            "max_position_embeddings": 256, "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0, "sliding_window": null,
            "tie_word_embeddings": false
        }"#,
    ),
    (
        "gemma",
        r#"{
            "model_type": "gemma",
            "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 2, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 8,
            "max_position_embeddings": 256, "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0
        }"#,
    ),
    (
        "phi3",
        r#"{
            "model_type": "phi3",
            "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 2, "num_attention_heads": 4,
            "num_key_value_heads": 4, "max_position_embeddings": 256,
            "rms_norm_eps": 1e-5, "rope_theta": 10000.0,
            "hidden_act": "silu", "tie_word_embeddings": false
        }"#,
    ),
    (
        "cohere",
        r#"{
            "model_type": "cohere",
            "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 2, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 8,
            "max_position_embeddings": 256, "layer_norm_eps": 1e-5,
            "rope_theta": 10000.0, "logit_scale": 0.0625,
            "use_sliding_window": false
        }"#,
    ),
    (
        "granite",
        r#"{
            "model_type": "granite",
            "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 2, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 8,
            "max_position_embeddings": 256, "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0
        }"#,
    ),
    (
        // Gemma 4 alternates five sliding layers to one full-attention layer,
        // and the two carry *different* KV geometry: the full ones use
        // `global_head_dim` with `num_global_key_value_heads`, the sliding ones
        // `head_dim` with `num_key_value_heads`. One cache serves both.
        "gemma4",
        r#"{
            "model_type": "gemma4_text",
            "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 6, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 8,
            "global_head_dim": 16, "num_global_key_value_heads": 1,
            "max_position_embeddings": 4096, "rms_norm_eps": 1e-6,
            "attention_k_eq_v": true, "tie_word_embeddings": true,
            "sliding_window": 64, "final_logit_softcapping": 30.0,
            "layer_types": [
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "sliding_attention", "full_attention"
            ],
            "rope_parameters": {
                "full_attention": {
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1000000.0,
                    "rope_type": "proportional"
                },
                "sliding_attention": { "rope_theta": 10000.0, "rope_type": "default" }
            },
            "num_kv_shared_layers": 0
        }"#,
    ),
];

fn ids(tokens: &[i32]) -> Array {
    Array::from_i32_slice_shaped(tokens, &[1, tokens.len() as i32])
}

/// The same invariant against a real checkpoint, which the synthetic configs
/// above cannot stand in for: released models have depths and per-layer
/// geometry the tiny ones do not, and a cache error that a two-layer toy
/// carries as a rounding difference compounds through forty-eight.
///
/// Checkpoints are not committed, so this is `#[ignore]`d and gated:
///
/// ```bash
/// PMETAL_CACHED_DECODE_MODEL=/path/to/checkpoint \
///     cargo test -p pmetal-models --test cached_decode -- --ignored --nocapture
/// ```
#[test]
#[ignore = "needs a real checkpoint; set PMETAL_CACHED_DECODE_MODEL"]
fn a_real_checkpoint_decodes_what_it_recomputes() {
    let Ok(dir) = std::env::var("PMETAL_CACHED_DECODE_MODEL") else {
        panic!("set PMETAL_CACHED_DECODE_MODEL to a checkpoint directory");
    };
    let mut model = DynamicModel::load(&dir).expect("checkpoint loads");
    let mut cache = model.create_cache(PROMPT.len() + STEPS + 1);

    let mut prefix = PROMPT.to_vec();
    let prefill = model
        .forward_with_cache(&ids(&prefix), None, Some(&mut cache))
        .expect("prefill");
    let mut cached_row = last_row(&prefill);

    for step in 0..STEPS {
        let fresh = model.forward(&ids(&prefix), None).expect("recompute");
        let fresh_row = last_row(&fresh);
        let scale = fresh_row
            .iter()
            .fold(0.0f32, |worst, v| worst.max(v.abs()))
            .max(1e-6);
        let diff = max_abs_diff(&cached_row, &fresh_row);
        println!(
            "step {step} over {} tokens: relative {:e}, cached argmax {} vs {}",
            prefix.len(),
            diff / scale,
            argmax(&cached_row),
            argmax(&fresh_row)
        );
        assert!(
            diff / scale < 1e-2,
            "step {step}: cached decode and uncached recompute disagree by {diff:e} \
             (relative {:e})",
            diff / scale
        );

        let next = argmax(&cached_row) as i32;
        prefix.push(next);
        let stepped = model
            .forward_with_cache(&ids(&[next]), None, Some(&mut cache))
            .expect("decode step");
        cached_row = last_row(&stepped);
    }
}

/// Logits at the last position, as f32.
fn last_row(logits: &Array) -> Vec<f32> {
    let shape = logits.shape();
    let (seq, vocab) = (shape[1], shape[2]);
    let mut row = logits.slice(&[0, seq - 1, 0], &[1, seq, vocab]);
    row.eval();
    row.to_f32_vec(vocab as usize).expect("to_f32_vec")
}

fn argmax(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |best, (index, value)| {
            if *value > best.1 {
                (index, *value)
            } else {
                best
            }
        })
        .0
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .fold(0.0f32, |worst, (x, y)| worst.max((x - y).abs()))
}

#[test]
fn a_cached_decode_reproduces_the_uncached_prefix() {
    for (name, config_json) in CASES {
        let mut model = DynamicModel::from_config(config_json).expect("architecture should build");
        let mut cache = model.create_cache(PROMPT.len() + STEPS + 1);

        let mut prefix = PROMPT.to_vec();
        let prefill = model
            .forward_with_cache(&ids(&prefix), None, Some(&mut cache))
            .expect("prefill");
        let mut cached_row = last_row(&prefill);

        for step in 0..STEPS {
            let fresh = model.forward(&ids(&prefix), None).expect("recompute");
            let fresh_row = last_row(&fresh);

            let scale = fresh_row
                .iter()
                .fold(0.0f32, |worst, v| worst.max(v.abs()))
                .max(1e-6);
            let diff = max_abs_diff(&cached_row, &fresh_row);
            assert!(
                diff / scale < 1e-3,
                "{name} step {step}: the cached decode and the uncached recompute of the \
                 same {} tokens disagree by {diff:e} (relative {:e}), so the cache is not \
                 reproducing the prefix",
                prefix.len(),
                diff / scale
            );
            assert_eq!(
                argmax(&cached_row),
                argmax(&fresh_row),
                "{name} step {step}: cached and uncached pick different tokens, which is \
                 what a degenerate generation looks like from the outside"
            );

            let next = argmax(&cached_row) as i32;
            prefix.push(next);
            let stepped = model
                .forward_with_cache(&ids(&[next]), None, Some(&mut cache))
                .expect("decode step");
            cached_row = last_row(&stepped);
        }
    }
}
