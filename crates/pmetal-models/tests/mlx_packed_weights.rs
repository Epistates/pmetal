//! An MLX-quantized checkpoint's weights stay packed on the model for inference.
//!
//! A decode step reads every weight once, so a 4-bit model unpacked to dense
//! reads four times the bytes. With `keep_quantized`, the words, scales and
//! biases go on the layers as they are and MLX's quantized matmul reads them.
//! Each test builds the MLX-quantized checkpoint of a tiny model from its own
//! weights, loads it both ways (packed, and unpacked to dense by the loader)
//! and holds the packed model to the dense one: they run the same math, so the
//! logits agree to float rounding, and every checkpoint tensor was consumed
//! (the loader fails otherwise).

mod common;

use std::collections::BTreeSet;

use common::mlx_quant::{
    Quantization, packed_modules, parameters, quantizable_modules, quantize_tensors,
    quantized_checkpoint, token_ids, write_checkpoint,
};
use pmetal_cache::test_utils::{ParityReport, Tolerance, argmax_last_axis};
use pmetal_models::DynamicModel;
use pmetal_models::dispatcher::DynamicModelLoadOptions;
use pmetal_runtime::compat::{ModuleParametersExt, random};
use serial_test::serial;

const PACKED: DynamicModelLoadOptions = DynamicModelLoadOptions {
    prefer_expert_offload: false,
    keep_quantized: true,
};
const UNPACKED: DynamicModelLoadOptions = DynamicModelLoadOptions {
    prefer_expert_offload: false,
    keep_quantized: false,
};

/// Bytes of the layer parameters (the embedding tables are compared apart).
fn layer_bytes(model: &DynamicModel) -> usize {
    model
        .flatten_params()
        .into_iter()
        .filter(|(k, _)| k.contains(".layers."))
        .map(|(_, v)| v.nbytes())
        .sum()
}

/// Build `config`'s model, quantize its checkpoint, load it packed and
/// unpacked, and hold the packed one to the dense one. `unpacked` names the
/// quantized modules that are allowed to come back dense, each a gap this test
/// records rather than hides.
fn check_arch(name: &str, config: &str, unpacked: &[&str]) -> DynamicModel {
    random::seed(7);
    let mut source = DynamicModel::from_config(config).expect("architecture builds");
    let (dir, quantized) = quantized_checkpoint(&mut source, config, Quantization::Q4);
    assert!(!quantized.is_empty(), "{name}: nothing was quantized");

    let mut packed = DynamicModel::load_with_options(dir.path(), PACKED)
        .unwrap_or_else(|e| panic!("{name}: packed load: {e}"));
    let mut dense = DynamicModel::load_with_options(dir.path(), UNPACKED)
        .unwrap_or_else(|e| panic!("{name}: unpacked load: {e}"));

    assert!(
        packed_modules(&mut dense).is_empty(),
        "{name}: the unpacking load kept something packed"
    );
    let want: BTreeSet<String> = quantized
        .iter()
        .filter(|m| !unpacked.iter().any(|u| m.as_str() == *u))
        .cloned()
        .collect();
    let got = packed_modules(&mut packed);
    assert_eq!(
        got.difference(&want).collect::<Vec<_>>(),
        Vec::<&String>::new(),
        "{name}: modules packed that the checkpoint left dense"
    );
    assert_eq!(
        want.difference(&got).collect::<Vec<_>>(),
        Vec::<&String>::new(),
        "{name}: quantized modules that were unpacked to dense"
    );

    let ids = token_ids(64, 12);
    let a = packed.forward(&ids, None).expect("packed forward");
    let b = dense.forward(&ids, None).expect("dense forward");
    pmetal_runtime::check_unobserved_error().expect("no runtime error");
    let report = ParityReport::compute(name, &a, &b, Tolerance::new(1e-3, 1e-3));
    assert!(
        report.passed(),
        "{name}: packed and dense logits differ, max |diff| {:.3e}",
        report.max_abs_diff
    );
    assert_eq!(argmax_last_axis(&a), argmax_last_axis(&b), "{name}");

    assert!(
        layer_bytes(&packed) * 2 < layer_bytes(&dense),
        "{name}: {} packed layer bytes against {} dense",
        layer_bytes(&packed),
        layer_bytes(&dense)
    );
    packed
}

macro_rules! dense_decoder {
    ($model_type:literal, $extra:literal) => {
        concat!(
            r#"{"model_type": ""#,
            $model_type,
            r#"", "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 8, "max_position_embeddings": 256, "rope_theta": 10000.0"#,
            $extra,
            "}"
        )
    };
}

#[test]
#[serial]
fn llama_stays_packed() {
    check_arch(
        "llama",
        dense_decoder!("llama", r#", "rms_norm_eps": 1e-6, "tie_word_embeddings": false"#),
        &[],
    );
}

#[test]
#[serial]
fn llama_with_a_tied_head_stays_packed() {
    check_arch(
        "llama tied",
        dense_decoder!("llama", r#", "rms_norm_eps": 1e-6, "tie_word_embeddings": true"#),
        &[],
    );
}

#[test]
#[serial]
fn qwen2_stays_packed() {
    check_arch(
        "qwen2",
        dense_decoder!("qwen2", r#", "rms_norm_eps": 1e-6, "tie_word_embeddings": false"#),
        &[],
    );
}

#[test]
#[serial]
fn qwen3_stays_packed() {
    check_arch(
        "qwen3",
        dense_decoder!("qwen3", r#", "rms_norm_eps": 1e-6, "tie_word_embeddings": true"#),
        &[],
    );
}

#[test]
#[serial]
fn mistral_stays_packed() {
    check_arch(
        "mistral",
        dense_decoder!(
            "mistral",
            r#", "rms_norm_eps": 1e-5, "sliding_window": null, "tie_word_embeddings": false"#
        ),
        &[],
    );
}

#[test]
#[serial]
fn cohere_stays_packed() {
    check_arch(
        "cohere",
        dense_decoder!(
            "cohere",
            r#", "layer_norm_eps": 1e-5, "logit_scale": 0.0625, "use_sliding_window": false"#
        ),
        &[],
    );
}

// Silence the helpers the later architecture tests use.
#[allow(dead_code)]
fn _unused() {
    let _ = (parameters, quantizable_modules, quantize_tensors, write_checkpoint);
}
