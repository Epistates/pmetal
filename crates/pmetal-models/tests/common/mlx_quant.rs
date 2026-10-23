//! Synthetic MLX-quantized checkpoints.
//!
//! An MLX converter writes a quantized module's tensors as siblings:
//! `x.weight` (the packed words), `x.scales` and, in affine mode, `x.biases`,
//! and records the width and group size in `config.json`'s `quantization`
//! block. [`quantize_tensors`] turns a model's own parameters into that layout
//! with the runtime's quantizer, so a test can hold the model that keeps the
//! words packed to the model that unpacked the same bytes to dense, which run
//! the same math.

#![allow(dead_code)]

use std::collections::{BTreeSet, HashMap};

use pmetal_models::DynamicModel;
use pmetal_runtime::compat::{Array, ModuleParametersExt, VisitLinears};

/// How a checkpoint is quantized.
#[derive(Debug, Clone, Copy)]
pub struct Quantization {
    pub bits: i32,
    pub group_size: i32,
}

impl Quantization {
    pub const Q4: Self = Self {
        bits: 4,
        group_size: 32,
    };
}

/// The module paths of a model whose weights an MLX converter quantizes: its
/// `Linear`s, embedding tables and stacked experts.
pub fn quantizable_modules(model: &mut DynamicModel) -> BTreeSet<String> {
    let mut modules = BTreeSet::new();
    model.visit_linears_mut("", &mut |path, _| {
        modules.insert(path.to_string());
    });
    model.visit_embeddings_mut("", &mut |path, _| {
        modules.insert(path.to_string());
    });
    model.visit_experts_mut("", 1, &mut |path, _, _| {
        modules.insert(path.to_string());
    });
    modules
}

/// A model's parameters by name.
pub fn parameters(model: &DynamicModel) -> Vec<(String, Array)> {
    let mut out: Vec<(String, Array)> = model
        .flatten_params()
        .into_iter()
        .map(|(name, array)| (name.to_string(), array))
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// Replace each `x.weight` whose module `x` is in `modules` (and whose last
/// axis the group size divides) with the packed words, `x.scales` and
/// `x.biases`. Everything else passes through. Returns the modules that were
/// quantized.
pub fn quantize_tensors(
    tensors: Vec<(String, Array)>,
    modules: &BTreeSet<String>,
    spec: Quantization,
) -> (Vec<(String, Array)>, BTreeSet<String>) {
    quantize_where(tensors, spec, |module, _| modules.contains(module))
}

/// [`quantize_tensors`] for the `x.weight` that `pick(x, weight)` selects.
pub fn quantize_where(
    tensors: Vec<(String, Array)>,
    spec: Quantization,
    pick: impl Fn(&str, &Array) -> bool,
) -> (Vec<(String, Array)>, BTreeSet<String>) {
    let mut out = Vec::with_capacity(tensors.len());
    let mut quantized = BTreeSet::new();
    for (name, value) in tensors {
        let module = name.strip_suffix(".weight").filter(|m| pick(m, &value));
        let eligible = module.is_some()
            && value.ndim() >= 2
            && value.dim(value.ndim() - 1) % spec.group_size == 0;
        match module.filter(|_| eligible) {
            Some(module) => {
                let (packed, scales, biases) = value.quantize_weights(spec.group_size, spec.bits);
                out.push((format!("{module}.weight"), packed));
                out.push((format!("{module}.scales"), scales));
                out.push((format!("{module}.biases"), biases));
                quantized.insert(module.to_string());
            }
            None => out.push((name, value)),
        }
    }
    (out, quantized)
}

/// `config` with the quantization block an MLX converter writes added.
pub fn config_with_quantization(config: &str, spec: Quantization) -> String {
    let mut json: serde_json::Value = serde_json::from_str(config).expect("config is JSON");
    json["quantization"] = serde_json::json!({
        "group_size": spec.group_size,
        "bits": spec.bits,
    });
    json.to_string()
}

/// Write `tensors` and `config` as a checkpoint directory.
pub fn write_checkpoint(tensors: &[(String, Array)], config: &str) -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    let entries: Vec<(&str, &Array)> = tensors.iter().map(|(k, v)| (k.as_str(), v)).collect();
    Array::save_safetensors(
        dir.path().join("model.safetensors").to_str().unwrap(),
        &entries,
    )
    .expect("save");
    std::fs::write(dir.path().join("config.json"), config).expect("config");
    dir
}

/// The MLX-quantized checkpoint of `model` as `config` builds it: every
/// quantizable module packed. Returns the directory and the modules packed.
pub fn quantized_checkpoint(
    model: &mut DynamicModel,
    config: &str,
    spec: Quantization,
) -> (tempfile::TempDir, BTreeSet<String>) {
    let modules = quantizable_modules(model);
    let (tensors, quantized) = quantize_tensors(parameters(model), &modules, spec);
    let dir = write_checkpoint(&tensors, &config_with_quantization(config, spec));
    (dir, quantized)
}

/// The modules of `model` that hold packed words after a load.
pub fn packed_modules(model: &mut DynamicModel) -> BTreeSet<String> {
    let mut packed = BTreeSet::new();
    model.visit_linears_mut("", &mut |path, linear| {
        if linear.quant.is_some() {
            packed.insert(path.to_string());
        }
    });
    model.visit_embeddings_mut("", &mut |path, embedding| {
        if embedding.quant.is_some() {
            packed.insert(path.to_string());
        }
    });
    model.visit_experts_mut("", 1, &mut |path, _, expert| {
        if expert.quant.is_some() {
            packed.insert(path.to_string());
        }
    });
    packed
}

/// Token ids for a forward pass.
pub fn token_ids(vocab: i32, len: i32) -> Array {
    let ids: Vec<i32> = (0..len).map(|i| (i * 7 + 3) % vocab).collect();
    Array::from_slice(&ids, &[1, len])
}

/// Per-name view, for assertions over a parameter map.
pub fn by_name(tensors: &[(String, Array)]) -> HashMap<&str, &Array> {
    tensors.iter().map(|(k, v)| (k.as_str(), v)).collect()
}
