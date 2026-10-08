//! Holds the Qwen4-Exp loader to the **released** checkpoints' layouts, from
//! their safetensors headers alone.
//!
//! `Qwen/Qwen3.8-Flash-Next` (bf16, fused experts), `-FP8` (per-expert block-FP8
//! experts, FP8 n-gram table) and `nvidia/...-NVFP4` (ModelOpt NVFP4 experts)
//! are 360 GB, 191 GB and 129 GB. What the loader decides on is in the headers:
//! every tensor's name, dtype and shape. So this test builds lazy placeholders
//! with exactly those dtypes and shapes (nothing is ever evaluated, so nothing
//! is allocated) and runs them through the production tensor path,
//! [`assign_qwen4_exp_tensors`]: sidecar unpacking, sanitization, strict
//! assignment into a model built from the released config. A tensor that maps
//! to no parameter, a parameter left unfilled, or a shape that disagrees fails
//! it. The n-gram shards and hash buffers, which the loader reads itself, are
//! checked against the config the same way.
//!
//! The headers and configs are not committed (the configs carry the models'
//! own licenses), so this is `#[ignore]`d and gated on a directory written by
//! `.strategy/parity/dump_qwen4_exp_headers_reference.py`:
//!
//! ```bash
//! PMETAL_QWEN4_EXP_HEADERS=/Volumes/AmBa/huggingface/qwen4-exp-headers \
//!     cargo test -p pmetal-models --test qwen4_exp_checkpoint_layout -- --ignored --nocapture
//! ```

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use pmetal_bridge::compat::{Array, Dtype};
use pmetal_models::architectures::qwen3_next::Qwen3NextRoutedExpertMode;
use pmetal_models::architectures::qwen4_exp::{
    CheckpointKeyRole, NgramHash, Qwen4ExpConfig, Qwen4ExpForCausalLM, Qwen4ExpLoadOptions,
    assign_qwen4_exp_tensors, checkpoint_key_role, ngram_shard_encoding,
};
use pmetal_models::dispatcher::ModelArchitecture;
use serial_test::serial;

const VARIANTS: [&str; 3] = [
    "Qwen_Qwen3.8-Flash-Next",
    "Qwen_Qwen3.8-Flash-Next-FP8",
    "nvidia_Qwen3.8-Flash-Next-NVFP4",
];

#[derive(Debug, serde::Deserialize)]
struct TensorInfo {
    dtype: String,
    shape: Vec<i64>,
}

/// A lazy placeholder with the tensor's dtype and shape, as MLX would load it
/// (E4M3 payloads arrive as `uint8`).
fn placeholder(info: &TensorInfo) -> Array {
    let dtype = match info.dtype.as_str() {
        "BF16" => Dtype::Bfloat16,
        "F16" => Dtype::Float16,
        "F32" => Dtype::Float32,
        "F8_E4M3" | "U8" => Dtype::Uint8,
        "I64" => Dtype::Int64,
        other => panic!("unexpected dtype {other}"),
    };
    let shape: Vec<i32> = info.shape.iter().map(|&d| d as i32).collect();
    Array::zeros(&shape, dtype.as_i32())
}

fn check_variant(dir: &Path) {
    assert_eq!(
        ModelArchitecture::detect(dir).expect("detects"),
        ModelArchitecture::Qwen4Exp,
        "{} is not detected as Qwen4-Exp",
        dir.display()
    );
    let wrapper: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let config = Qwen4ExpConfig::from_json(&wrapper["text_config"].to_string())
        .expect("released config parses and validates");
    let tensors: HashMap<String, TensorInfo> =
        serde_json::from_str(&std::fs::read_to_string(dir.join("tensors.json")).unwrap()).unwrap();

    let mut weights = HashMap::new();
    let mut skipped: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut shards: BTreeMap<usize, Vec<(&String, &TensorInfo)>> = BTreeMap::new();
    let mut scaled: Vec<usize> = Vec::new();
    let mut hash_buffers = 0;
    for (name, info) in &tensors {
        match checkpoint_key_role(name) {
            CheckpointKeyRole::Weight => {
                weights.insert(name.clone(), placeholder(info));
            }
            CheckpointKeyRole::NgramTable { layer } => {
                shards.entry(layer).or_default().push((name, info));
            }
            CheckpointKeyRole::NgramScale { layer } => {
                assert_eq!(
                    info.shape.iter().product::<i64>(),
                    1,
                    "{name} is per-tensor"
                );
                scaled.push(layer);
            }
            CheckpointKeyRole::HashBuffer { .. } => {
                assert_eq!(info.dtype, "I64", "{name}");
                let expected = if name.ends_with("layer_multipliers") {
                    config.ngram_size as i64
                } else {
                    config.ngram_heads() as i64
                };
                assert_eq!(info.shape, vec![expected], "{name}");
                hash_buffers += 1;
            }
            CheckpointKeyRole::Skipped(reason) => *skipped.entry(reason).or_default() += 1,
        }
    }
    let weight_count = weights.len();

    let mut model =
        Qwen4ExpForCausalLM::new_for_loading(config.clone(), Qwen3NextRoutedExpertMode::Resident)
            .expect("model builds from the released config");
    let assigned = assign_qwen4_exp_tensors(&mut model, weights, Qwen4ExpLoadOptions::default())
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    // NVFP4 experts stay on packed kernels; everything else loads dense.
    let packed = model
        .model
        .layers
        .iter()
        .filter(|l| l.mlp.packed_experts.is_some())
        .count();
    let expect_packed = if dir.to_string_lossy().contains("NVFP4") {
        config.num_hidden_layers as usize
    } else {
        0
    };
    assert_eq!(packed, expect_packed, "layers with packed experts");

    // The n-gram tables: one per PLE layer, split as the config says, rows
    // and width matching the hash.
    let ple_layers: Vec<usize> = config
        .ple_layer_ids
        .iter()
        .map(|&id| id as usize - 1)
        .collect();
    assert_eq!(shards.keys().copied().collect::<Vec<_>>(), ple_layers);
    for (k, &layer) in ple_layers.iter().enumerate() {
        let parts = &shards[&layer];
        assert_eq!(parts.len(), config.split_ngram_parts as usize);
        let scale = scaled.contains(&layer).then_some(1.0);
        let rows: i64 = parts
            .iter()
            .map(|(name, info)| {
                let shape: Vec<u64> = info.shape.iter().map(|&d| d as u64).collect();
                ngram_shard_encoding(&config, name, &info.dtype, &shape, scale)
                    .unwrap_or_else(|e| panic!("{name}: {e}"));
                info.shape[0]
            })
            .sum();
        assert_eq!(rows, NgramHash::new(&config, k).rows, "layer {layer} rows");
    }

    println!(
        "{}: {} tensors | {weight_count} weights and sidecars -> {assigned} parameters and \
         packed stacks ({packed} layers packed) | {} n-gram shards ({} with an FP8 scale) | \
         {hash_buffers} hash buffers | skipped {skipped:?}",
        dir.file_name().unwrap().to_string_lossy(),
        tensors.len(),
        shards.values().map(Vec::len).sum::<usize>(),
        scaled.len(),
    );
}

#[test]
#[ignore = "needs PMETAL_QWEN4_EXP_HEADERS (see the module docs)"]
#[serial]
fn released_qwen4_exp_layouts_load_completely() {
    let root = std::env::var("PMETAL_QWEN4_EXP_HEADERS")
        .expect("set PMETAL_QWEN4_EXP_HEADERS to the dumper's output directory");
    for variant in VARIANTS {
        check_variant(&Path::new(&root).join(variant));
    }
}
