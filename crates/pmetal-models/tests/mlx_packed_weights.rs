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
use pmetal_runtime::compat::{Array, Dtype, ModuleParametersExt, VisitLinears, random};
use serial_test::serial;

const PACKED: DynamicModelLoadOptions = DynamicModelLoadOptions {
    prefer_expert_offload: false,
    keep_quantized: true,
};
const UNPACKED: DynamicModelLoadOptions = DynamicModelLoadOptions {
    prefer_expert_offload: false,
    keep_quantized: false,
};

/// `vocab_size` of a config, at the top level or under `text_config`.
fn vocab_size(config: &str) -> i32 {
    let json: serde_json::Value = serde_json::from_str(config).expect("config is JSON");
    json.get("vocab_size")
        .or_else(|| json.get("text_config").and_then(|t| t.get("vocab_size")))
        .and_then(serde_json::Value::as_i64)
        .expect("vocab_size") as i32
}

/// A committed fixture's config.
fn fixture_config(name: &str) -> String {
    std::fs::read_to_string(common::fixture_path(name)).expect("fixture config")
}

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

    let ids = token_ids(vocab_size(config), 12);
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
        dense_decoder!(
            "llama",
            r#", "rms_norm_eps": 1e-6, "tie_word_embeddings": false"#
        ),
        &[],
    );
}

#[test]
#[serial]
fn llama_with_a_tied_head_stays_packed() {
    check_arch(
        "llama tied",
        dense_decoder!(
            "llama",
            r#", "rms_norm_eps": 1e-6, "tie_word_embeddings": true"#
        ),
        &[],
    );
}

#[test]
#[serial]
fn qwen2_stays_packed() {
    check_arch(
        "qwen2",
        dense_decoder!(
            "qwen2",
            r#", "rms_norm_eps": 1e-6, "tie_word_embeddings": false"#
        ),
        &[],
    );
}

#[test]
#[serial]
fn qwen3_stays_packed() {
    check_arch(
        "qwen3",
        dense_decoder!(
            "qwen3",
            r#", "rms_norm_eps": 1e-6, "tie_word_embeddings": true"#
        ),
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

#[test]
#[serial]
fn qwen3_moe_keeps_its_experts_packed() {
    let model = check_arch("qwen3_moe", &fixture_config("qwen3_moe_config.json"), &[]);
    assert_experts_packed(model, 3 * 3);
}

#[test]
#[serial]
fn deepseek_keeps_its_experts_packed() {
    // The routed experts' down projection (16 inputs) and the low-rank query
    // up-projection (48) do not divide into a group: the checkpoint leaves them
    // dense and so does the loader.
    let model = check_arch(
        "deepseek_v3",
        &fixture_config("deepseek_native_v3_config.json"),
        &[],
    );
    assert_experts_packed(model, 2 * 2);
}

#[test]
#[serial]
fn nemotron_h_keeps_its_experts_packed() {
    // Two sparse layers, each with an up and a down projection stacked.
    let model = check_arch(
        "nemotron_h",
        &fixture_config("nemotron_h_hybrid_moe_config.json"),
        &[],
    );
    assert_experts_packed(model, 2 * 2);
}

/// The stacked experts hold packed words with their scales: `count` projections.
fn assert_experts_packed(mut model: DynamicModel, count: usize) {
    let mut seen = 0;
    model.visit_experts_mut("", 1, &mut |path, _, expert| {
        if expert.quant.is_some() {
            assert_eq!(expert.weight.dtype(), Dtype::Uint32, "{path}");
            seen += 1;
        }
    });
    assert_eq!(seen, count, "stacked expert projections kept packed");
}

/// A committed fixture's weights, in name order.
fn fixture_tensors(name: &str) -> Vec<(String, Array)> {
    let mut tensors: Vec<_> = common::load_shard(&common::fixture_path(name))
        .into_iter()
        .collect();
    tensors.sort_by(|a, b| a.0.cmp(&b.0));
    tensors
}

/// Quantize a fixture checkpoint's `quantized` modules as an MLX converter
/// would, write it with `config` and load it packed and unpacked. Returns the
/// two models and the quantized module names (in the checkpoint's spelling).
fn load_both(
    config: &str,
    tensors: Vec<(String, Array)>,
    pick: impl Fn(&str) -> bool,
) -> (DynamicModel, DynamicModel, BTreeSet<String>) {
    let spec = Quantization::Q4;
    let (tensors, quantized) =
        common::mlx_quant::quantize_where(tensors, spec, |module, _| pick(module));
    let dir = write_checkpoint(
        &tensors,
        &common::mlx_quant::config_with_quantization(config, spec),
    );
    let packed = DynamicModel::load_with_options(dir.path(), PACKED).expect("packed load");
    let dense = DynamicModel::load_with_options(dir.path(), UNPACKED).expect("unpacked load");
    (packed, dense, quantized)
}

/// The packed model's logits against the dense one's, from `ids`.
fn assert_same_logits(name: &str, packed: &mut DynamicModel, dense: &mut DynamicModel, vocab: i32) {
    let ids = token_ids(vocab, 12);
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
}

/// The Qwen3.5 fixtures in an MLX conversion's layout: the language model under
/// `language_model.model.`, the vision tower and the multi-token predictor
/// dropped, and the routed experts stacked one tensor per projection under
/// `mlp.switch_mlp`.
fn qwen3_5_in_mlx_layout(weights: &str) -> Vec<(String, Array)> {
    use pmetal_runtime::compat::ops::slice_axis;
    let mut out = Vec::new();
    for (key, value) in fixture_tensors(weights) {
        if key.starts_with("mtp.") || key.starts_with("model.visual.") {
            continue;
        }
        let key = match key.strip_prefix("model.language_model.") {
            Some(rest) => format!("language_model.model.{rest}"),
            None if key == "lm_head.weight" => "language_model.lm_head.weight".to_string(),
            None => key,
        };
        if let Some(layer) = key.strip_suffix(".mlp.experts.gate_up_proj") {
            let half = value.dim(1) / 2;
            out.push((
                format!("{layer}.mlp.switch_mlp.gate_proj.weight"),
                slice_axis(&value, 1, 0, half),
            ));
            out.push((
                format!("{layer}.mlp.switch_mlp.up_proj.weight"),
                slice_axis(&value, 1, half, 2 * half),
            ));
        } else if let Some(layer) = key.strip_suffix(".mlp.experts.down_proj") {
            out.push((format!("{layer}.mlp.switch_mlp.down_proj.weight"), value));
        } else {
            out.push((key, value));
        }
    }
    out
}

/// What an MLX converter quantizes in a Qwen3.5 checkpoint.
fn qwen3_5_quantizes(module: &str) -> bool {
    module.starts_with("language_model.")
        && (module.ends_with("_proj")
            || module.contains(".in_proj_")
            || module.ends_with(".mlp.gate")
            || module.ends_with("shared_expert_gate")
            || module.ends_with("embed_tokens"))
        || module == "language_model.lm_head"
}

/// A checkpoint module's path in the model's tree.
fn qwen3_5_model_path(module: &str) -> String {
    match module.strip_prefix("language_model.model.") {
        Some(rest) => format!("model.{rest}"),
        None => module.replace("language_model.", ""),
    }
}

fn check_qwen3_5(name: &str, config: &str, weights: &str, sparse: bool) {
    random::seed(7);
    let config = fixture_config(config);
    let (mut packed, mut dense, quantized) =
        load_both(&config, qwen3_5_in_mlx_layout(weights), qwen3_5_quantizes);

    let (experts, rest): (Vec<_>, Vec<_>) =
        quantized.iter().partition(|m| m.contains(".switch_mlp."));
    let want: BTreeSet<String> = rest.iter().map(|m| qwen3_5_model_path(m)).collect();
    assert_eq!(
        packed_modules(&mut packed),
        want,
        "{name}: projections and tables"
    );
    assert!(packed_modules(&mut dense).is_empty(), "{name}");

    // The routed experts of every sparse layer run from packed words, and a
    // dense load leaves none.
    let packed_layers = |model: &DynamicModel| -> usize {
        let DynamicModel::Qwen3Next(model) = model else {
            panic!("{name}: not a Qwen3.5 model");
        };
        model
            .model
            .layers
            .iter()
            .filter(|layer| {
                matches!(
                    &layer.mlp,
                    pmetal_models::architectures::qwen3_next::Qwen3NextFeedForward::MoE(block)
                        if block.packed_experts.is_some()
                )
            })
            .count()
    };
    if sparse {
        assert!(!experts.is_empty(), "{name}: no experts were quantized");
        assert_eq!(packed_layers(&packed), experts.len() / 3, "{name}");
    }
    assert_eq!(packed_layers(&dense), 0, "{name}");

    assert_same_logits(name, &mut packed, &mut dense, vocab_size(&config));
}

#[test]
#[serial]
fn qwen3_5_dense_stays_packed() {
    check_qwen3_5(
        "qwen3_5",
        "qwen3_5_swish_config.json",
        "qwen3_5_swish_weights.safetensors",
        false,
    );
}

#[test]
#[serial]
fn qwen3_5_sparse_keeps_its_experts_packed() {
    check_qwen3_5(
        "qwen3_5_moe",
        "qwen3_5_moe_config.json",
        "qwen3_5_moe_weights.safetensors",
        true,
    );
}

/// A Gemma 4 fixture in an MLX conversion's layout: the language model under
/// `language_model.model.`, the experts stacked as `experts.switch_glu`.
fn gemma4_in_mlx_layout(weights: &str) -> Vec<(String, Array)> {
    use pmetal_runtime::compat::ops::slice_axis;
    let mut out = Vec::new();
    for (key, value) in fixture_tensors(weights) {
        let key = match key.strip_prefix("model.") {
            Some(rest) => format!("language_model.model.{rest}"),
            None => key,
        };
        if let Some(layer) = key.strip_suffix(".experts.gate_up_proj") {
            let half = value.dim(1) / 2;
            out.push((
                format!("{layer}.experts.switch_glu.gate_proj.weight"),
                slice_axis(&value, 1, 0, half),
            ));
            out.push((
                format!("{layer}.experts.switch_glu.up_proj.weight"),
                slice_axis(&value, 1, half, 2 * half),
            ));
        } else if let Some(layer) = key.strip_suffix(".experts.down_proj") {
            out.push((
                format!("{layer}.experts.switch_glu.down_proj.weight"),
                value,
            ));
        } else {
            out.push((key, value));
        }
    }
    out
}

fn gemma4_quantizes(module: &str) -> bool {
    module.ends_with("_proj")
        || module.ends_with(".router.proj")
        || module.ends_with("embed_tokens")
        || module.ends_with("embed_tokens_per_layer")
        || module.ends_with("per_layer_input_gate")
        || module.ends_with("per_layer_model_projection")
}

/// A checkpoint module's path in the model's tree.
fn gemma4_model_path(module: &str) -> String {
    module
        .replace("language_model.model.", "model.")
        .replace(".router.", ".moe.router.")
        .replace(".per_layer_input_gate", ".per_layer_input_block.gate_proj")
        .replace(".per_layer_projection", ".per_layer_input_block.projection")
        .replace(
            "model.per_layer_model_projection",
            "model.per_layer_inputs.model_projection",
        )
        .replace(
            "model.embed_tokens_per_layer",
            "model.per_layer_inputs.embed_tokens",
        )
}

fn check_gemma4(name: &str, config: &str, weights: &str, expert_layers: usize) {
    random::seed(7);
    let (mut packed, mut dense, quantized) =
        load_both(config, gemma4_in_mlx_layout(weights), gemma4_quantizes);

    let (experts, rest): (Vec<_>, Vec<_>) = quantized
        .iter()
        .partition(|m| m.contains(".experts.switch_glu."));
    let want: BTreeSet<String> = rest.iter().map(|m| gemma4_model_path(m)).collect();
    assert_eq!(
        packed_modules(&mut packed),
        want,
        "{name}: projections and tables"
    );
    assert!(packed_modules(&mut dense).is_empty(), "{name}");

    let packed_layers = |model: &DynamicModel| -> usize {
        let DynamicModel::Gemma4(model) = model else {
            panic!("{name}: not a Gemma 4 model");
        };
        model
            .model
            .layers
            .iter()
            .filter(|layer| {
                layer
                    .moe
                    .as_ref()
                    .is_some_and(|moe| moe.experts.is_packed())
            })
            .count()
    };
    assert_eq!(
        experts.len(),
        3 * expert_layers,
        "{name}: experts quantized"
    );
    assert_eq!(packed_layers(&packed), expert_layers, "{name}");
    assert_eq!(packed_layers(&dense), 0, "{name}");

    assert_same_logits(name, &mut packed, &mut dense, vocab_size(config));
}

#[test]
#[serial]
fn gemma4_dense_stays_packed() {
    check_gemma4(
        "gemma4",
        common::gemma4::synthetic_config_json(),
        "gemma4_synth_weights.safetensors",
        0,
    );
}

#[test]
#[serial]
fn gemma4_with_per_layer_inputs_stays_packed() {
    check_gemma4(
        "gemma4 shared",
        common::gemma4::shared_synthetic_config_json(),
        "gemma4_shared_synth_weights.safetensors",
        0,
    );
}

#[test]
#[serial]
fn gemma4_moe_keeps_its_experts_packed() {
    check_gemma4(
        "gemma4 moe",
        common::gemma4::moe_synthetic_config_json(),
        "gemma4_moe_synth_weights.safetensors",
        2,
    );
}

// Silence the helpers the later architecture tests use.
#[allow(dead_code)]
fn _unused() {
    let _ = (
        parameters,
        quantizable_modules,
        quantize_tensors,
        write_checkpoint,
    );
}
