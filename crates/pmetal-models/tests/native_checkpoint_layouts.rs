//! The Qwen3.5-family native loader reads one model out of every layout a
//! checkpoint of it ships in.
//!
//! The base is the small fp32 routed-experts model `qwen3_5_moe` (8 experts,
//! 2 per token, a shared expert behind its own gate, GatedDeltaNet and full
//! attention layers) as transformers saves it: experts fused into
//! `gate_up_proj` and `down_proj`. The same weights are rewritten into the
//! other layouts, and every one must load and give the base's logits:
//!
//! * one module per expert (`experts.{e}.gate_proj.weight`, transformers 4),
//! * MLX's own: `language_model.model.*` keys and the experts stacked as
//!   `switch_mlp.{gate,up,down}_proj`,
//! * MLX's, quantized the way MLX's converter quantizes it: every
//!   projection, the embedding and the head at the file's bits and group size
//!   (4-bit affine, or mxfp4), and `mlp.gate` and `shared_expert_gate` at 8
//!   bits and group size 64 through the model's `quant_predicate`. Against the
//!   same checkpoint with every packed weight dequantized to dense.

mod common;

use std::collections::HashMap;
use std::path::Path;

use common::{fixture_path, load_shard, ref_tensor};
use pmetal_cache::test_utils::max_abs_diff;
use pmetal_runtime::compat::{Array, ops};
use pmetal_runtime::qwen3_native::{NativeCache, forward_step, load_config, load_model};
use pmetal_runtime::{InlineArray, QuantizedMode};
use serde_json::{Value, json};

type Tensors = Vec<(String, Array)>;

const EXPERTS: i32 = 8;
const GROUP_SIZE: i32 = 32;

fn drain(op: &str) {
    pmetal_runtime::check_unobserved_error().unwrap_or_else(|e| panic!("{op}: bridge error: {e}"));
}

fn base_tensors() -> HashMap<String, Array> {
    load_shard(&fixture_path("qwen3_5_moe_weights.safetensors"))
}

fn base_config() -> Value {
    let text = std::fs::read_to_string(fixture_path("qwen3_5_moe_config.json")).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn input_ids() -> Array {
    let reference = load_shard(&fixture_path("qwen3_5_moe_reference.safetensors"));
    ref_tensor(&reference, "input_ids").clone()
}

fn write_checkpoint(config: &Value, tensors: &Tensors) -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        dir.path().join("config.json"),
        serde_json::to_string_pretty(config).unwrap(),
    )
    .unwrap();
    let entries: Vec<(&str, &Array)> = tensors.iter().map(|(k, v)| (k.as_str(), v)).collect();
    InlineArray::save_safetensors(
        dir.path().join("model.safetensors").to_str().unwrap(),
        &entries,
    )
    .expect("save the weights");
    dir
}

/// Every position's logits of a forward over the fixture's prompt.
fn logits(dir: &Path) -> Array {
    let config = load_config(dir).expect("config parses");
    let weights = load_model(dir, &config).expect("checkpoint loads");
    drain("load");
    let mut cache = NativeCache::new_empty(&weights);
    let out = forward_step(&weights, &input_ids(), &mut cache);
    drain("forward");
    out
}

/// The logits of the base checkpoint, in transformers' own layout.
fn baseline() -> Array {
    let tensors: Tensors = base_tensors().into_iter().collect();
    let dir = write_checkpoint(&base_config(), &tensors);
    logits(dir.path())
}

fn experts_key(key: &str, name: &str) -> Option<String> {
    key.strip_suffix(name)
        .map(|prefix| prefix.trim_end_matches('.').to_string())
}

/// `[E, 2I, H]` fused gate and up into `[E, I, H]` each.
fn split_gate_up(gate_up: &Array) -> (Array, Array) {
    let inter = gate_up.dim(1) / 2;
    (
        ops::slice_axis(gate_up, 1, 0, inter),
        ops::slice_axis(gate_up, 1, inter, 2 * inter),
    )
}

/// Expert `e` of a `[E, out, in]` stack, `[out, in]`.
fn expert(stack: &Array, e: i32) -> Array {
    let (out, input) = (stack.dim(1), stack.dim(2));
    ops::slice_axis(stack, 0, e, e + 1).reshape(&[out, input])
}

fn without_vision(tensors: &HashMap<String, Array>) -> impl Iterator<Item = (&String, &Array)> {
    tensors
        .iter()
        .filter(|(k, _)| !k.starts_with("model.visual."))
}

/// One module per expert, under the same `model.language_model.` prefix.
fn per_expert_layout() -> Tensors {
    let mut out = Tensors::new();
    for (key, value) in without_vision(&base_tensors()) {
        if let Some(prefix) = experts_key(key, "experts.gate_up_proj") {
            let (gate, up) = split_gate_up(value);
            for e in 0..EXPERTS {
                out.push((
                    format!("{prefix}.experts.{e}.gate_proj.weight"),
                    expert(&gate, e),
                ));
                out.push((
                    format!("{prefix}.experts.{e}.up_proj.weight"),
                    expert(&up, e),
                ));
            }
        } else if let Some(prefix) = experts_key(key, "experts.down_proj") {
            for e in 0..EXPERTS {
                out.push((
                    format!("{prefix}.experts.{e}.down_proj.weight"),
                    expert(value, e),
                ));
            }
        } else {
            out.push((key.clone(), value.clone()));
        }
    }
    out
}

/// How MLX's converter quantizes one module.
#[derive(Clone, Copy)]
struct Spec {
    bits: i32,
    group_size: i32,
    mode: QuantizedMode,
}

/// A tensor packed with `spec`, and its dense equivalent.
struct Packed {
    weight: Array,
    scales: Array,
    biases: Option<Array>,
    dense: Array,
}

fn pack(dense: &Array, spec: Spec) -> Packed {
    match spec.mode {
        QuantizedMode::Affine => {
            let (weight, scales, biases) = dense.quantize_weights(spec.group_size, spec.bits);
            let dense = weight.dequantize(&scales, &biases, spec.group_size, spec.bits);
            Packed {
                weight,
                scales,
                biases: Some(biases),
                dense,
            }
        }
        mode => {
            let (weight, scales) = dense.quantize_weights_mode(spec.group_size, spec.bits, mode);
            let dense = weight.dequantize_mode(&scales, None, spec.group_size, spec.bits, mode);
            Packed {
                weight,
                scales,
                biases: None,
                dense,
            }
        }
    }
}

/// The checkpoint MLX's converter would write for the base model, and the base
/// weights it stands for: the same tensors in transformers' layout, each
/// quantized one replaced by its dequantization.
///
/// `file` is the file-level quantization (`None`: dense). `mlp.gate` and
/// `shared_expert_gate` are quantized at 8 bits, group size 64, affine.
fn mlx_layout(file: Option<Spec>) -> (Tensors, Tensors, Value) {
    let (mut mlx, mut oracle) = (Tensors::new(), Tensors::new());
    let mut block = serde_json::Map::new();
    if let Some(spec) = file {
        block.insert("group_size".into(), spec.group_size.into());
        block.insert("bits".into(), spec.bits.into());
        if spec.mode == QuantizedMode::Mxfp4 {
            block.insert("mode".into(), "mxfp4".into());
        }
    }
    let gates = Spec {
        bits: 8,
        group_size: 64,
        mode: QuantizedMode::Affine,
    };
    let mlx_name = |key: &str| {
        if let Some(rest) = key.strip_prefix("model.language_model.") {
            format!("language_model.model.{rest}")
        } else {
            format!("language_model.{key}")
        }
    };

    for (key, value) in without_vision(&base_tensors()) {
        // The module a `.weight` belongs to, and the spec it is quantized
        // with: the predicate's for the two gates, the file's for any other
        // matrix, none for norms, the convolution and the GatedDeltaNet decay.
        let module = key.strip_suffix(".weight");
        let spec = module.and_then(|module| {
            if module.ends_with("mlp.gate") || module.ends_with("shared_expert_gate") {
                file.map(|_| gates)
            } else if value.ndim() == 2 && !module.contains("conv1d") && !module.contains("norm") {
                file
            } else {
                None
            }
        });

        if let Some(prefix) = experts_key(key, "experts.gate_up_proj") {
            let (gate, up) = split_gate_up(value);
            let down_key = format!("{prefix}.experts.down_proj");
            let down = &base_tensors()[&down_key];
            let mut dense = Vec::new();
            for (name, tensor) in [("gate_proj", &gate), ("up_proj", &up), ("down_proj", down)] {
                let module = mlx_name(&format!("{prefix}.switch_mlp.{name}"));
                match file {
                    Some(spec) => {
                        let p = pack(tensor, spec);
                        mlx.push((format!("{module}.weight"), p.weight));
                        mlx.push((format!("{module}.scales"), p.scales));
                        if let Some(b) = p.biases {
                            mlx.push((format!("{module}.biases"), b));
                        }
                        dense.push(p.dense);
                    }
                    None => {
                        mlx.push((format!("{module}.weight"), tensor.clone()));
                        dense.push(tensor.clone());
                    }
                }
            }
            oracle.push((
                format!("{prefix}.experts.gate_up_proj"),
                ops::concatenate_axis(&[&dense[0], &dense[1]], 1),
            ));
            oracle.push((down_key, dense.remove(2)));
            continue;
        }
        if experts_key(key, "experts.down_proj").is_some() {
            continue; // written with its gate_up_proj
        }

        match (module, spec) {
            (Some(module), Some(spec)) => {
                let p = pack(value, spec);
                let name = mlx_name(module);
                mlx.push((format!("{name}.weight"), p.weight));
                mlx.push((format!("{name}.scales"), p.scales));
                if let Some(b) = p.biases {
                    mlx.push((format!("{name}.biases"), b));
                }
                oracle.push((key.clone(), p.dense));
                // Only a module that differs from the file's block needs an entry.
                if spec.bits != file.map_or(0, |f| f.bits)
                    || spec.group_size != file.map_or(0, |f| f.group_size)
                    || spec.mode != file.map_or(QuantizedMode::Affine, |f| f.mode)
                {
                    block.insert(
                        name,
                        json!({"group_size": spec.group_size, "bits": spec.bits}),
                    );
                }
            }
            _ => {
                mlx.push((mlx_name(key), value.clone()));
                oracle.push((key.clone(), value.clone()));
            }
        }
    }
    (mlx, oracle, Value::Object(block))
}

fn with_quantization(mut config: Value, block: &Value) -> Value {
    if block.as_object().is_some_and(|b| !b.is_empty()) {
        config["quantization"] = block.clone();
        config["quantization_config"] = block.clone();
    }
    config
}

fn assert_same_logits(what: &str, got: &Array, want: &Array, tolerance: f32) {
    let worst = max_abs_diff(got, want);
    println!("{what}: max |logit difference| {worst:e}");
    assert!(
        worst <= tolerance,
        "{what}: logits differ from the reference by {worst:e} (limit {tolerance:e})"
    );
}

/// A bf16 checkpoint stays bf16 through a multi-token pass and a decode step:
/// the logits and every cache (KV and the GatedDeltaNet convolution and
/// recurrent state, which carries its own dtype), for the hybrid routed model.
#[test]
fn a_bf16_checkpoint_stays_bf16_through_prefill_and_decode() {
    use pmetal_runtime::compat::Dtype;
    use pmetal_runtime::decode::prompt_tokens_to_input;

    let bf16 = Dtype::Bfloat16.as_i32();
    let tensors: Tensors = base_tensors()
        .into_iter()
        .map(|(k, v)| {
            let cast = if v.ndim() > 0 { v.as_dtype(bf16) } else { v };
            (k, cast)
        })
        .collect();
    let dir = write_checkpoint(&base_config(), &tensors);
    let config = load_config(dir.path()).expect("config parses");
    let weights = load_model(dir.path(), &config).expect("checkpoint loads");
    assert_eq!(weights.model_dtype, bf16);

    let mut cache = NativeCache::new_empty(&weights);
    let ids = input_ids();
    let logits = forward_step(&weights, &ids, &mut cache);
    let step = forward_step(&weights, &prompt_tokens_to_input(&[3]), &mut cache);
    drain("bf16 forward");
    assert_eq!(logits.dtype_raw(), bf16, "prefill logits");
    assert_eq!(step.dtype_raw(), bf16, "decode logits");
    for (i, kv) in cache.kv_caches.iter().enumerate() {
        assert_eq!(
            kv.keys.as_ref().unwrap().dtype_raw(),
            bf16,
            "layer {i} keys"
        );
        assert_eq!(
            kv.values.as_ref().unwrap().dtype_raw(),
            bf16,
            "layer {i} values"
        );
    }
    for (i, gdn) in cache.gdn_caches.iter().enumerate() {
        assert_eq!(
            gdn.conv_state.as_ref().unwrap().dtype_raw(),
            bf16,
            "gdn {i} conv"
        );
    }
}

/// transformers 4 and NVIDIA ModelOpt store one module per expert.
#[test]
fn one_module_per_expert_loads_the_same_model() {
    let want = baseline();
    let dir = write_checkpoint(&base_config(), &per_expert_layout());
    assert_same_logits("per-expert layout", &logits(dir.path()), &want, 1e-5);
}

/// What MLX's converter writes for an unquantized model: `language_model`
/// keys and the experts stacked as `switch_mlp`, `[E, out, in]`.
#[test]
fn mlx_stacked_experts_load_the_same_model() {
    let want = baseline();
    let (mlx, _, block) = mlx_layout(None);
    let dir = write_checkpoint(&with_quantization(base_config(), &block), &mlx);
    assert_same_logits("mlx stacked layout", &logits(dir.path()), &want, 1e-5);
}

fn quantized_checkpoint_matches_its_dense_equivalent(file: Spec, label: &str, tolerance: f32) {
    let (mlx, oracle, block) = mlx_layout(Some(file));
    // The router and the shared expert's gate are packed in what mlx writes.
    assert!(
        mlx.iter()
            .any(|(k, _)| k.ends_with("layers.0.mlp.gate.scales"))
            && mlx
                .iter()
                .any(|(k, _)| k.ends_with("layers.0.mlp.shared_expert_gate.scales")),
        "the checkpoint under test packs both gates"
    );

    let dense = write_checkpoint(&base_config(), &oracle);
    let packed = write_checkpoint(&with_quantization(base_config(), &block), &mlx);
    assert_same_logits(
        label,
        &logits(packed.path()),
        &logits(dense.path()),
        tolerance,
    );
}

/// 4-bit affine, group size 32, with the gates at 8 bits and group size 64.
#[test]
fn mlx_affine_checkpoint_matches_its_dequantized_weights() {
    quantized_checkpoint_matches_its_dense_equivalent(
        Spec {
            bits: 4,
            group_size: GROUP_SIZE,
            mode: QuantizedMode::Affine,
        },
        "mlx 4-bit affine",
        5e-4,
    );
}

/// mxfp4: scales alone, no biases; the gates stay affine at 8 bits.
#[test]
fn mlx_mxfp4_checkpoint_matches_its_dequantized_weights() {
    quantized_checkpoint_matches_its_dense_equivalent(
        Spec {
            bits: 4,
            group_size: GROUP_SIZE,
            mode: QuantizedMode::Mxfp4,
        },
        "mlx mxfp4",
        5e-4,
    );
}
