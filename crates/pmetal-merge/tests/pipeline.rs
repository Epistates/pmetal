//! End-to-end behaviour of the merge pipelines on small synthetic models:
//! layer-sliced frankenmerges, vocabulary mismatches, base-model handling and
//! the output layout.

mod common;

use common::{assert_close, read_config, read_tensor, write_model_with_config};
use pmetal_merge::{MergeConfig, MergeError, VocabMismatch, run_merge, run_merge_sliced};
use safetensors::Dtype;
use std::path::Path;

/// A 4-layer model: layer `l` tensors hold `base + l`, so an output value
/// says which source layer it came from. Also a vision tower with a
/// `layers.0` of its own, an embedding, a final norm and a head.
fn layered_model(dir: &Path, base: f32) {
    let mut tensors: Vec<(String, Vec<f32>, Vec<usize>)> = Vec::new();
    for layer in 0..4 {
        let value = base + layer as f32;
        tensors.push((
            format!("model.layers.{layer}.mlp.up_proj.weight"),
            vec![value; 2],
            vec![2],
        ));
        tensors.push((
            format!("model.layers.{layer}.input_layernorm.weight"),
            vec![value + 0.5],
            vec![1],
        ));
    }
    tensors.push((
        "model.vision_tower.layers.0.proj.weight".into(),
        vec![base * 100.0],
        vec![1],
    ));
    tensors.push(("model.embed_tokens.weight".into(), vec![base * 10.0; 2], vec![2]));
    tensors.push(("model.norm.weight".into(), vec![base * 20.0], vec![1]));
    tensors.push(("lm_head.weight".into(), vec![base * 30.0; 2], vec![2]));
    let specs: Vec<common::Spec<'_>> = tensors
        .iter()
        .map(|(n, v, s)| (n.as_str(), v.clone(), s.clone()))
        .collect();
    write_model_with_config(
        dir,
        &specs,
        Dtype::F32,
        r#"{"num_hidden_layers": 4, "hidden_size": 2,
            "layer_types": ["full", "sliding", "sliding", "full"]}"#,
    );
}

fn sliced_config(method: &str, yaml_tail: &str, a: &Path, b: &Path, out: &Path) -> MergeConfig {
    MergeConfig::from_yaml(&format!(
        "merge_method: {method}\ndtype: float32\noutput_path: {out}\n{yaml_tail}",
        out = out.display()
    )
    .replace("$A", &a.display().to_string())
    .replace("$B", &b.display().to_string()))
    .unwrap()
}

#[test]
fn sliced_merge_stacks_layers_and_takes_the_head_from_the_last_slice() {
    let work = tempfile::tempdir().unwrap();
    let (a, b, out) = (work.path().join("a"), work.path().join("b"), work.path().join("out"));
    layered_model(&a, 1.0); // layer l holds 1 + l
    layered_model(&b, 100.0); // layer l holds 100 + l

    // Output layers: A1, A2, B0, B1.
    let config = sliced_config(
        "passthrough",
        "slices:\n  - sources:\n      - model: $A\n        layer_range: [1, 3]\n  - sources:\n      - model: $B\n        layer_range: [0, 2]\n",
        &a,
        &b,
        &out,
    );
    run_merge(&config).unwrap();

    let up = |l: usize| read_tensor(&out, &format!("model.layers.{l}.mlp.up_proj.weight"));
    assert_close(&up(0), &[2.0; 2], 0.0);
    assert_close(&up(1), &[3.0; 2], 0.0);
    assert_close(&up(2), &[100.0; 2], 0.0);
    assert_close(&up(3), &[101.0; 2], 0.0);
    // Norm and head follow the last slice; the embedding the first.
    assert_close(&read_tensor(&out, "model.embed_tokens.weight"), &[10.0; 2], 0.0);
    assert_close(&read_tensor(&out, "model.norm.weight"), &[2000.0], 0.0);
    assert_close(&read_tensor(&out, "lm_head.weight"), &[3000.0; 2], 0.0);
    // The vision tower's own `layers.0` is not part of the language stack.
    assert_close(
        &read_tensor(&out, "model.vision_tower.layers.0.proj.weight"),
        &[100.0],
        0.0,
    );

    // The config describes the 4 layers actually written, with the layer
    // types of the source layers they were taken from (A1, A2, B0, B1).
    let cfg = read_config(&out);
    assert_eq!(cfg["num_hidden_layers"], 4);
    assert_eq!(
        cfg["layer_types"],
        serde_json::json!(["sliding", "sliding", "full", "sliding"])
    );
}

#[test]
fn sliced_task_arithmetic_reads_the_base_at_the_source_layer() {
    let work = tempfile::tempdir().unwrap();
    let (a, b, base, out) = (
        work.path().join("a"),
        work.path().join("b"),
        work.path().join("base"),
        work.path().join("out"),
    );
    layered_model(&a, 10.0); // layer l holds 10 + l
    layered_model(&b, 20.0); // layer l holds 20 + l
    layered_model(&base, 0.0); // layer l holds l

    // One output layer, merging A's layer 2 with B's layer 0. Their base is
    // the layer the first source reads: base layer 2 (value 2), not base
    // layer 0 (the output index).
    let mut config = sliced_config(
        "task_arithmetic",
        "slices:\n  - sources:\n      - model: $A\n        layer_range: [2, 3]\n      - model: $B\n        layer_range: [0, 1]\n",
        &a,
        &b,
        &out,
    );
    config.base_model = Some(base.display().to_string());
    run_merge_sliced(&config).unwrap();

    // tau_A = 12 - 2 = 10, tau_B = 20 - 2 = 18, merged = 2 + 10 + 18 = 30.
    assert_close(
        &read_tensor(&out, "model.layers.0.mlp.up_proj.weight"),
        &[30.0; 2],
        1e-6,
    );
}

#[test]
fn a_slice_naming_layers_the_model_lacks_is_an_error() {
    let work = tempfile::tempdir().unwrap();
    let (a, b, out) = (work.path().join("a"), work.path().join("b"), work.path().join("out"));
    layered_model(&a, 1.0);
    layered_model(&b, 2.0);
    let config = sliced_config(
        "passthrough",
        "slices:\n  - sources:\n      - model: $A\n        layer_range: [2, 6]\n",
        &a,
        &b,
        &out,
    );
    let err = run_merge(&config).unwrap_err().to_string();
    assert!(err.contains("layer 4"), "{err}");
}

fn vocab_model(dir: &Path, rows: usize, fill: f32) {
    let embed: Vec<f32> = (0..rows).flat_map(|r| [fill + r as f32, fill]).collect();
    write_model_with_config(
        dir,
        &[
            ("model.embed_tokens.weight", embed.clone(), vec![rows, 2]),
            ("lm_head.weight", embed, vec![rows, 2]),
        ],
        Dtype::F32,
        &format!(r#"{{"vocab_size": {rows}}}"#),
    );
}

fn vocab_config(policy: &str, models: &[&Path], base: Option<&Path>, out: &Path) -> MergeConfig {
    let mut yaml = format!(
        "merge_method: {}\ndtype: float32\noutput_path: {}\nvocab_mismatch: {policy}\nmodels:\n",
        if base.is_some() { "task_arithmetic" } else { "linear" },
        out.display()
    );
    for m in models {
        yaml.push_str(&format!("  - model: {}\n", m.display()));
    }
    let mut config = MergeConfig::from_yaml(&yaml).unwrap();
    config.base_model = base.map(|b| b.display().to_string());
    config
}

#[test]
fn differing_vocabulary_sizes_are_an_error_by_default() {
    let work = tempfile::tempdir().unwrap();
    let (a, b) = (work.path().join("a"), work.path().join("b"));
    vocab_model(&a, 3, 0.0);
    vocab_model(&b, 5, 0.0);
    let config = vocab_config("error", &[&a, &b], None, &work.path().join("out"));
    let err = run_merge(&config).unwrap_err();
    assert!(
        matches!(err, MergeError::InvalidConfig(ref m) if m.contains("vocab_mismatch")),
        "{err}"
    );
    assert_eq!(config.vocab_mismatch, VocabMismatch::Error);
}

#[test]
fn truncate_cuts_to_the_smallest_vocabulary_and_updates_the_config() {
    let work = tempfile::tempdir().unwrap();
    let (a, b, out) = (work.path().join("a"), work.path().join("b"), work.path().join("out"));
    vocab_model(&a, 3, 0.0);
    vocab_model(&b, 5, 10.0);
    run_merge(&vocab_config("truncate", &[&a, &b], None, &out)).unwrap();

    // Rows 0..3 of both, averaged: row r is ((r, 0) + (10 + r, 10)) / 2.
    assert_close(
        &read_tensor(&out, "model.embed_tokens.weight"),
        &[5.0, 5.0, 6.0, 5.0, 7.0, 5.0],
        1e-6,
    );
    assert_eq!(read_config(&out)["vocab_size"], 3);
}

#[test]
fn extend_grows_to_the_largest_vocabulary_filling_rows_from_the_model_that_has_them() {
    let work = tempfile::tempdir().unwrap();
    let (a, b, out) = (work.path().join("a"), work.path().join("b"), work.path().join("out"));
    vocab_model(&a, 3, 0.0);
    vocab_model(&b, 5, 10.0);
    run_merge(&vocab_config("extend", &[&a, &b], None, &out)).unwrap();

    // Rows 0..3 average the two models; rows 3 and 4 exist only in B, and A
    // is padded with B's rows, so they come out as B's.
    assert_close(
        &read_tensor(&out, "model.embed_tokens.weight"),
        &[5.0, 5.0, 6.0, 5.0, 7.0, 5.0, 13.0, 10.0, 14.0, 10.0],
        1e-6,
    );
    assert_eq!(read_config(&out)["vocab_size"], 5);
}

#[test]
fn extend_with_a_base_pads_the_missing_rows_as_no_change() {
    let work = tempfile::tempdir().unwrap();
    let (a, b, base, out) = (
        work.path().join("a"),
        work.path().join("b"),
        work.path().join("base"),
        work.path().join("out"),
    );
    vocab_model(&base, 3, 0.0); // rows (0,0) (1,0) (2,0)
    vocab_model(&a, 3, 1.0); // task vector (+1, +1) on every row
    vocab_model(&b, 5, 0.0); // two extra rows, and no change on the first three
    run_merge(&vocab_config("extend", &[&a, &b], Some(&base), &out)).unwrap();

    // Task arithmetic over rows 0..3: base + tau_A + tau_B = base + (1, 1).
    // Rows 3 and 4 exist only in B and the base is padded with them, so
    // neither model changes them: B's own rows.
    assert_close(
        &read_tensor(&out, "model.embed_tokens.weight"),
        &[1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 3.0, 0.0, 4.0, 0.0],
        1e-6,
    );
}

#[test]
fn a_tensor_missing_from_the_base_is_named_in_the_error() {
    let work = tempfile::tempdir().unwrap();
    let (a, base, out) = (work.path().join("a"), work.path().join("base"), work.path().join("out"));
    write_model_with_config(&a, &[("x", vec![1.0], vec![1]), ("y", vec![2.0], vec![1])], Dtype::F32, "{}");
    write_model_with_config(&base, &[("x", vec![0.0], vec![1])], Dtype::F32, "{}");
    let mut config = vocab_config("error", &[&a], Some(&base), &out);
    config.vocab_mismatch = VocabMismatch::Error;
    let err = run_merge(&config).unwrap_err().to_string();
    assert!(err.contains("y") && err.contains("base"), "{err}");
}

#[test]
fn shard_size_splits_the_output_and_writes_the_index() {
    let work = tempfile::tempdir().unwrap();
    let (a, out) = (work.path().join("a"), work.path().join("out"));
    let tensors: Vec<(String, Vec<f32>, Vec<usize>)> = (0..6)
        .map(|i| (format!("t{i}"), vec![i as f32; 4], vec![4]))
        .collect();
    let specs: Vec<common::Spec<'_>> = tensors
        .iter()
        .map(|(n, v, s)| (n.as_str(), v.clone(), s.clone()))
        .collect();
    write_model_with_config(&a, &specs, Dtype::F32, "{}");

    let mut config = vocab_config("error", &[&a], None, &out);
    config.max_shard_size = Some(32); // two 16-byte tensors per shard
    run_merge(&config).unwrap();

    let index: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(out.join("model.safetensors.index.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(index["metadata"]["total_size"], 96);
    assert_eq!(index["weight_map"].as_object().unwrap().len(), 6);
    for i in 0..6 {
        assert_close(&read_tensor(&out, &format!("t{i}")), &[i as f32; 4], 0.0);
    }
    assert!(out.join("model-00001-of-00003.safetensors").is_file());
}

#[test]
fn union_tokenizers_are_rejected_instead_of_silently_ignored() {
    let work = tempfile::tempdir().unwrap();
    let (a, out) = (work.path().join("a"), work.path().join("out"));
    write_model_with_config(&a, &[("x", vec![1.0], vec![1])], Dtype::F32, "{}");
    let mut config = vocab_config("error", &[&a], None, &out);
    config.tokenizer = Some(pmetal_merge::TokenizerConfig {
        source: "union".into(),
    });
    let err = run_merge(&config).unwrap_err().to_string();
    assert!(err.contains("union"), "{err}");
}
