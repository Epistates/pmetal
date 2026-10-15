//! Model loading utilities for PMetal.
//!
//! Provides functionality to load model weights from safetensor files,
//! with support for HuggingFace model formats and weight name mapping.
use pmetal_runtime::compat::{Array, ModuleParameters, ModuleParametersExt};

use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::architectures::utils::LoadReport;

mod diffusion;
mod families;

pub use diffusion::*;
pub use families::*;

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("SafeTensors error: {0}")]
    SafeTensors(String),
    #[error("Missing weight: {0}")]
    MissingWeight(String),
    #[error("Shape mismatch for {key}: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        key: String,
        expected: Vec<i32>,
        actual: Vec<i32>,
    },
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("MLX error: {0}")]
    Mlx(String),
    #[error("MLX IO error: {0}")]
    MlxIo(#[from] pmetal_runtime::compat::IoError),
}

impl From<pmetal_runtime::compat::Exception> for LoadError {
    fn from(e: pmetal_runtime::compat::Exception) -> Self {
        Self::Mlx(e.to_string())
    }
}

const PARAM_EVAL_BATCH_SIZE: usize = 128;

/// Read one safetensors file into a name → array map, as stored (no
/// dequantization or renaming).
pub fn load_safetensors_file(
    path: impl AsRef<std::path::Path>,
) -> Result<HashMap<String, Array>, LoadError> {
    load_shard(path.as_ref())
}

/// Load a safetensors shard file into a `HashMap<String, Array>`.
///
/// This wraps `pmetal_runtime::inline_array::load_safetensors_shard` which
/// returns `Option<Vec<(String, InlineArray)>>` and converts it into the
/// `HashMap<String, Array>` that the model loaders expect.
fn load_shard(path: &std::path::Path) -> Result<HashMap<String, Array>, LoadError> {
    let path_str = path.to_str().ok_or_else(|| {
        LoadError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("Non-UTF8 path: {:?}", path),
        ))
    })?;
    pmetal_runtime::inline_array::load_safetensors_shard(path_str)
        .map(|pairs| pairs.into_iter().collect::<HashMap<_, _>>())
        .ok_or_else(|| LoadError::SafeTensors(format!("Failed to load safetensors: {}", path_str)))
}

/// Width of one quantized tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TensorQuant {
    bits: i32,
    group_size: i32,
}

#[derive(Debug, Clone)]
struct MlxQuantizationConfig {
    bits: i32,
    group_size: i32,
    /// Per-tensor widths, keyed by the tensor's own checkpoint key.
    overrides: HashMap<String, TensorQuant>,
}

impl MlxQuantizationConfig {
    /// The width to unpack `key` at, which is the model-wide one unless the
    /// checkpoint said otherwise for this tensor.
    fn for_tensor(&self, key: &str) -> TensorQuant {
        self.overrides.get(key).copied().unwrap_or(TensorQuant {
            bits: self.bits,
            group_size: self.group_size,
        })
    }
}

/// How many checkpoint keys [`detect_namespace_prefix`] probes before deciding.
/// Each probe narrows the candidate set, and one alone is enough whenever the
/// parameter tree has no repeated tails.
const NAMESPACE_PROBE_KEYS: usize = 8;

/// The namespace shift, if any, that reconciles a checkpoint's keys with a
/// model's parameter names.
///
/// Publishers disagree on whether the text tower carries its wrapper's
/// namespace. `Qwen/Qwen3-Embedding-8B` ships `embed_tokens.weight`,
/// `layers.0.…` and `norm.weight` where `Qwen3ForCausalLM` holds all three
/// under `model.`, so an exact-name load matches nothing and the model answers
/// from its random init.
///
/// The shift is detected once for the whole checkpoint rather than per key. A
/// per-key fallback would let a checkpoint carrying both `layers.0.x` and
/// `model.layers.0.x` resolve both onto the same parameter, in whatever order
/// the `HashMap` happened to yield them.
///
/// Returns `None` when the names already line up, when nothing explains them,
/// or when two prefixes explain them equally well — which leaves the mismatch
/// for [`LoadReport`] to report rather than papering over it with a guess.
fn detect_namespace_prefix(
    params: &HashMap<String, &mut Array>,
    loaded: &HashMap<String, Array>,
) -> Option<String> {
    let explains = |prefix: &str| -> usize {
        loaded
            .keys()
            .filter(|key| params.contains_key(&format!("{prefix}{key}")))
            .count()
    };

    let already = explains("");
    if already == loaded.len() {
        return None;
    }

    // A candidate is whatever the parameter tree puts in front of a checkpoint
    // key. A handful of probes surfaces every prefix in play; scoring then runs
    // each candidate against the whole checkpoint, so one stray key among the
    // probes cannot veto a prefix that explains all the rest.
    let mut probes: Vec<&String> = loaded.keys().collect();
    probes.sort_unstable();
    probes.truncate(NAMESPACE_PROBE_KEYS);

    let mut candidates: HashSet<String> = HashSet::new();
    for probe in probes {
        let tail = format!(".{probe}");
        candidates.extend(
            params
                .keys()
                .filter_map(|name| name.strip_suffix(tail.as_str()))
                .map(|prefix| format!("{prefix}.")),
        );
    }

    let mut ranked: Vec<(usize, String)> = candidates
        .into_iter()
        .map(|prefix| (explains(&prefix), prefix))
        .collect();
    ranked.sort_unstable();

    match ranked.pop() {
        // A prefix that explains no more than the bare names is not a shift,
        // and two prefixes explaining equally much is a coin flip that would
        // silently half-load the model.
        Some((best, prefix)) if best > already && !ranked.iter().any(|(n, _)| *n == best) => {
            Some(prefix)
        }
        _ => None,
    }
}

/// Assign every checkpoint tensor whose name matches a parameter path,
/// reporting the ones that matched nothing.
///
/// Matching is by exact name, after [`detect_namespace_prefix`] reconciles a
/// checkpoint that omits the model's namespace. A checkpoint laid out
/// differently in some other way still contributes nothing and leaves the model
/// on its random init, so the unmatched keys come back in the report —
/// `load_generic_weights_renamed` logs a summary, and a caller that knows the
/// checkpoint should map completely can assert on it.
fn assign_loaded_weights<M: ModuleParameters + ModuleParametersExt>(
    model: &mut M,
    loaded: HashMap<String, Array>,
) -> LoadReport {
    let mut params = model.flatten_params_mut();
    let prefix = detect_namespace_prefix(&params, &loaded);
    let mut report = LoadReport::default();
    for (key, value) in loaded {
        let name = match &prefix {
            Some(prefix) => format!("{prefix}{key}"),
            None => key.clone(),
        };
        match params.get_mut(&name) {
            Some(param) => {
                **param = value;
                report.loaded += 1;
            }
            // The checkpoint's own spelling, not the probed one, since that is
            // what the reader has in front of them.
            None => report.skipped.push(key),
        }
    }
    report
}

fn eval_loaded_parameters<M: ModuleParameters + ModuleParametersExt>(
    model: &M,
) -> Result<(), LoadError> {
    let params = model.flatten_params();
    let arrays: Vec<Array> = params.into_values().collect();
    for chunk in arrays.chunks(PARAM_EVAL_BATCH_SIZE) {
        pmetal_runtime::compat::transforms::eval(chunk.iter())
            .map_err(|e| LoadError::Mlx(e.to_string()))?;
    }
    Ok(())
}

fn load_mlx_quantization_config(
    model_dir: &Path,
) -> Result<Option<MlxQuantizationConfig>, LoadError> {
    let config_path = model_dir.join("config.json");
    if !config_path.exists() {
        return Ok(None);
    }

    let raw = std::fs::read_to_string(&config_path)?;
    // Through `config_value`, not `serde_json::from_str`: Python's `json.dump`
    // emits bare `Infinity`, which is not JSON. `nvidia/Nemotron-H-8B-Base-8K`
    // ships `"time_step_limit": [0.0, Infinity]`, and reading it strictly here
    // failed the *whole load* with a parse error 46 lines from anything this
    // function cares about. Architecture detection already routed around it;
    // this was the other half, and no config-only sweep could see it because
    // no config-only sweep loads weights.
    let json =
        crate::dispatcher::config_value(&raw).map_err(|e| LoadError::SafeTensors(e.to_string()))?;
    let Some(quant) = json
        .get("quantization")
        .or_else(|| json.get("quantization_config"))
        .or_else(|| {
            json.get("text_config").and_then(|text| {
                text.get("quantization")
                    .or_else(|| text.get("quantization_config"))
            })
        })
    else {
        return Ok(None);
    };

    let bits = quant
        .get("bits")
        .and_then(|value| value.as_i64())
        .unwrap_or(4) as i32;
    let group_size = quant
        .get("group_size")
        .and_then(|value| value.as_i64())
        .unwrap_or(64) as i32;
    if bits <= 0 {
        return Err(LoadError::SafeTensors(format!(
            "Invalid MLX quantization bits in config.json: {bits}"
        )));
    }
    if group_size <= 0 {
        return Err(LoadError::SafeTensors(format!(
            "Invalid MLX quantization group_size in config.json: {group_size}"
        )));
    }

    let mut overrides: HashMap<String, TensorQuant> = HashMap::new();

    // MLX's own layout, and the one mlx-community QAT releases ship: each
    // module that differs from the model-wide setting gets its own entry
    // *beside* `bits` and `group_size`, keyed by module path, e.g.
    //
    //   "language_model.model.layers.0.mlp.gate_proj": {"bits": 8, "group_size": 64}
    //
    // MLX-format loaders read these when rebuilding the model. Not
    // reading them unpacks those tensors at the model-wide width, which yields
    // the right byte count and the wrong shape, so the failure surfaces several
    // ops later as a norm complaining about its input.
    //
    // A module may also map to `false`, meaning it was left unquantized; those
    // carry no `.scales`/`.biases` and are skipped on that basis below.
    if let Some(object) = quant.as_object() {
        for (module_path, value) in object {
            let Some(entry) = value.as_object() else {
                continue;
            };
            let read = |field: &str, fallback: i32| {
                entry
                    .get(field)
                    .and_then(|v| v.as_i64())
                    .map(|v| v as i32)
                    .unwrap_or(fallback)
            };
            let quant = TensorQuant {
                bits: read("bits", bits),
                group_size: read("group_size", group_size),
            };
            if quant.bits <= 0 || quant.group_size <= 0 {
                return Err(LoadError::SafeTensors(format!(
                    "Invalid MLX quantization for {module_path}: {quant:?}"
                )));
            }
            // Keyed by module; the packed tensor is that module's `weight`.
            overrides.insert(format!("{module_path}.weight"), quant);
        }
    }

    // pmetal's own mixed-precision output, which records bits per *tensor* key.
    if let Some(object) = quant
        .get("per_tensor_overrides")
        .and_then(|value| value.as_object())
    {
        for (tensor_key, value) in object {
            if let Some(per_tensor_bits) = value.as_i64() {
                overrides.insert(
                    tensor_key.clone(),
                    TensorQuant {
                        bits: per_tensor_bits as i32,
                        group_size,
                    },
                );
            }
        }
    }

    Ok(Some(MlxQuantizationConfig {
        bits,
        group_size,
        overrides,
    }))
}

fn quant_aux_base_key(key: &str) -> Option<&str> {
    key.strip_suffix(".scales")
        .or_else(|| key.strip_suffix(".biases"))
}

fn is_quant_aux_key(key: &str) -> bool {
    quant_aux_base_key(key).is_some()
}

/// The packed tensor a `.scales` / `.biases` entry belongs to, if it is present.
///
/// ⚠️ Two layouts ship, and they differ by one path segment.
///
/// **MLX**, and so every `mlx-community` release, stores a quantized module's
/// three parameters as *siblings*: `mlp.up_proj.weight` holds the packed
/// payload next to `mlp.up_proj.scales` and `mlp.up_proj.biases`.
///
/// **pmetal's own quantizer** suffixes the full tensor name instead, so the
/// same module comes out as `mlp.up_proj.weight` with
/// `mlp.up_proj.weight.scales` and `mlp.up_proj.weight.biases`.
///
/// Preferring the `.weight` child and falling back to the base name lands on
/// the right tensor either way. A module path is never itself a tensor, so the
/// two cases cannot both apply to one entry.
fn packed_weight_key_for(aux_key: &str, weights: &HashMap<String, Array>) -> Option<String> {
    let base = quant_aux_base_key(aux_key)?;
    let child = format!("{base}.weight");
    if weights.contains_key(&child) {
        Some(child)
    } else if weights.contains_key(base) {
        Some(base.to_string())
    } else {
        None
    }
}

fn dequantize_mlx_quantized_weights(
    weights: &mut HashMap<String, Array>,
    config: &MlxQuantizationConfig,
) {
    // `(packed, scales, biases)` triples, resolved before anything is replaced
    // so the lookups above see the checkpoint as it was read.
    let triples: Vec<(String, String, String)> = weights
        .keys()
        .filter(|key| key.ends_with(".scales"))
        .filter_map(|scales_key| {
            let weight_key = packed_weight_key_for(scales_key, weights)?;
            let biases_key = format!("{}.biases", quant_aux_base_key(scales_key)?);
            weights
                .contains_key(&biases_key)
                .then(|| (weight_key, scales_key.clone(), biases_key))
        })
        .collect();

    for (weight_key, scales_key, biases_key) in triples {
        let (Some(weight), Some(scales), Some(biases)) = (
            weights.get(&weight_key).cloned(),
            weights.get(&scales_key).cloned(),
            weights.get(&biases_key).cloned(),
        ) else {
            continue;
        };
        // Both the width *and* the group size can vary per tensor. Unpacking a
        // mixed-precision checkpoint at one model-wide setting silently
        // produces wrongly-shaped weights.
        let quant = config.for_tensor(&weight_key);
        let dequantized = weight.dequantize(&scales, &biases, quant.group_size, quant.bits);
        weights.insert(weight_key, dequantized);
    }

    weights.retain(|key, _| !is_quant_aux_key(key));
}

/// Unpack every per-tensor-sidecar scheme to dense, as MLX's own format is
/// unpacked above: NVIDIA ModelOpt nvfp4 and fp8, and Qwen's block-scaled FP8.
///
/// ⚠️ Before this, their sidecars fell through as unmatched keys with a
/// warning, and the packed bytes were assigned to dense `Linear` weights. The
/// load reported every weight matched, and anything trained on it trained on
/// noise.
pub(crate) fn dequantize_sidecar_weights(
    weights: &mut HashMap<String, Array>,
) -> Result<(), LoadError> {
    use pmetal_runtime::native_loader as nl;

    if !weights
        .keys()
        .any(|key| nl::quant_sidecar_base(key).is_some())
    {
        return Ok(());
    }
    let dtype = pmetal_runtime::native_weight::detect_model_dtype(|key| {
        weights.get(key).map(|w| w.dtype_raw())
    });

    for (base, weight) in nl::take_modelopt_weights(weights).map_err(LoadError::SafeTensors)? {
        let dense = weight
            .to_dense(dtype)
            .map_err(|e| LoadError::SafeTensors(format!("{base}.weight: {e}")))?;
        weights.insert(format!("{base}.weight"), dense);
    }

    let block_fp8: Vec<String> = weights
        .keys()
        .filter_map(|key| key.strip_suffix(".weight_scale_inv"))
        .map(ToOwned::to_owned)
        .collect();
    for base in block_fp8 {
        let weight_key = format!("{base}.weight");
        let (Some(scale_inv), Some(weight)) = (
            weights.remove(&format!("{base}.weight_scale_inv")),
            weights.remove(&weight_key),
        ) else {
            return Err(LoadError::SafeTensors(format!(
                "FP8 scale {base}.weight_scale_inv has no weight"
            )));
        };
        let dense = nl::dequantize_fp8_e4m3_scaled_weight(&weight, &scale_inv, dtype)
            .map_err(|e| LoadError::SafeTensors(format!("{weight_key}: {e}")))?;
        weights.insert(weight_key, dense);
    }

    // An activation calibration with no weight scale beside it changes no
    // weight, so it is safe to drop. A weight scale is not.
    weights.retain(|key, _| !key.ends_with(".input_scale"));
    let leftover: Vec<&String> = weights
        .keys()
        .filter(|key| nl::quant_sidecar_base(key).is_some())
        .take(10)
        .collect();
    if !leftover.is_empty() {
        return Err(LoadError::SafeTensors(format!(
            "quantization scales this loader cannot apply, so their weights would load as raw bytes: {leftover:?}"
        )));
    }
    Ok(())
}

/// Whether a sharded checkpoint carries per-tensor quantization sidecars, which
/// cannot be applied a shard at a time.
fn index_has_quant_sidecars(index: &WeightIndex) -> bool {
    index
        .weight_map
        .keys()
        .any(|key| pmetal_runtime::native_loader::quant_sidecar_base(key).is_some())
}

/// Weight index for sharded models.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct WeightIndex {
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
    pub weight_map: HashMap<String, String>,
}

/// Validate that a shard path does not escape the model directory (path traversal protection).
///
/// HuggingFace cache uses symlinks: snapshot files point to `../../blobs/`.
/// We validate that the canonical shard path stays within the HF repo root
/// (the common ancestor of both `snapshots/` and `blobs/`), not just the
/// snapshot directory itself.
pub(crate) fn validate_shard_path(
    model_dir: &Path,
    shard_file: &str,
) -> Result<std::path::PathBuf, LoadError> {
    // Reject path traversal in the shard filename itself
    if shard_file.contains("..") || shard_file.starts_with('/') {
        return Err(LoadError::Io(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            format!("Shard filename contains path traversal: {}", shard_file),
        )));
    }
    let shard_path = model_dir.join(shard_file);
    let canonical_dir = model_dir.canonicalize().map_err(LoadError::Io)?;
    let canonical_shard = shard_path.canonicalize().map_err(|e| {
        LoadError::Io(std::io::Error::new(
            e.kind(),
            format!(
                "Shard file not found: {} (in {})",
                shard_file,
                model_dir.display()
            ),
        ))
    })?;
    // First check: shard is directly inside model_dir (non-symlinked case)
    if canonical_shard.starts_with(&canonical_dir) {
        // Return the original path, so the `.safetensors` extension is kept
        return Ok(shard_path);
    }
    // Second check: HF cache layout — shard symlinks to ../../blobs/ within
    // the same repo directory (e.g. models--Org--Name/{snapshots,blobs}/)
    // Allow if both canonical paths share the same HF repo root.
    if let Some(repo_root) = find_hf_repo_root(&canonical_dir) {
        if canonical_shard.starts_with(&repo_root) {
            // Return original symlink path to preserve .safetensors extension
            return Ok(shard_path);
        }
        // Third check: huggingface_hub's cache-wide shared blob store, where a
        // repo's `blobs/<etag>` is itself a symlink to `<cache>/blobs/<xx>/<hash>`
        // so Xet files are stored once across repos. Still inside the cache.
        if let Some(shared_blobs) = repo_root.parent().map(|cache| cache.join("blobs")) {
            if canonical_shard.starts_with(&shared_blobs) {
                return Ok(shard_path);
            }
        }
    }
    Err(LoadError::Io(std::io::Error::new(
        std::io::ErrorKind::PermissionDenied,
        format!("Shard path escapes model directory: {:?}", shard_path),
    )))
}

/// Find the HuggingFace repo root directory for a given path.
///
/// HF cache layout: `~/.cache/huggingface/hub/models--Org--Name/snapshots/<hash>/`
/// The repo root is `models--Org--Name/` which contains both `snapshots/` and `blobs/`.
fn find_hf_repo_root(path: &Path) -> Option<std::path::PathBuf> {
    let mut current = Some(path);
    while let Some(p) = current {
        if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
            if name.starts_with("models--") || name.starts_with("datasets--") {
                return Some(p.to_path_buf());
            }
        }
        current = p.parent();
    }
    None
}

/// Load generic weights using safetensors.
pub fn load_generic_weights<M: ModuleParameters + ModuleParametersExt>(
    model: &mut M,
    model_dir: impl AsRef<Path>,
) -> Result<(), LoadError> {
    load_generic_weights_renamed(model, model_dir, |_| None)
}

/// [`load_generic_weights`] with a hook that rewrites checkpoint keys into
/// pmetal parameter paths.
///
/// `rename` returns `Some(path)` for a key that needs rewriting and `None` to
/// pass it through. This exists for architectures whose parameter tree is
/// shaped differently from the checkpoint but not differently enough to justify
/// a bespoke loader — the alternative is `assign_loaded_weights` silently
/// dropping every renamed tensor, since it matches by exact name.
pub fn load_generic_weights_renamed<M: ModuleParameters + ModuleParametersExt>(
    model: &mut M,
    model_dir: impl AsRef<Path>,
    rename: impl Fn(&str) -> Option<String>,
) -> Result<(), LoadError> {
    let apply = |loaded: HashMap<String, Array>| -> HashMap<String, Array> {
        loaded
            .into_iter()
            .map(|(key, value)| match rename(&key) {
                Some(renamed) => (renamed, value),
                None => (key, value),
            })
            .collect()
    };

    let model_dir = model_dir.as_ref();
    let mut report = LoadReport::default();

    // ⚠️ A quantized checkpoint cannot be streamed a shard at a time. Unpacking
    // needs `{module}.weight` together with its `.scales` and `.biases`, which
    // are three separate tensors, and nothing in the format promises a shard
    // boundary won't fall between them. Assemble it whole, unpack, then assign.
    //
    // Without this every architecture reaching the model through
    // `load_generic_weights` — llama, qwen2, mistral, gemma, phi, cohere and
    // the rest of the `simple_load!` arms — handed the packed `uint32` payload
    // straight to the model. The first norm then reported a weight that did not
    // match the width of its input, several ops downstream of the real problem.
    if load_mlx_quantization_config(model_dir)?.is_some() {
        let loaded = apply(load_weights(model_dir)?);
        report += assign_loaded_weights(model, loaded);
        eval_loaded_parameters(model)?;
        report.log_summary(model_dir);
        return Ok(());
    }

    // One file is read whole either way; `load_weights` also unpacks any
    // per-tensor quantization it carries.
    let single_file = model_dir.join("model.safetensors");
    if single_file.exists() {
        let loaded = apply(load_weights(model_dir)?);
        report += assign_loaded_weights(model, loaded);
        eval_loaded_parameters(model)?;
        report.log_summary(model_dir);
        return Ok(());
    }
    let index_path = model_dir.join("model.safetensors.index.json");
    if !index_path.exists() {
        return Err(LoadError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "No weights found",
        )));
    }
    let index_content = std::fs::read_to_string(&index_path).map_err(LoadError::Io)?;
    let index: WeightIndex = serde_json::from_str(&index_content)?;
    // Same reason as the MLX case above: a scale and its weight can sit in
    // different shards.
    if index_has_quant_sidecars(&index) {
        let loaded = apply(load_weights(model_dir)?);
        report += assign_loaded_weights(model, loaded);
        eval_loaded_parameters(model)?;
        report.log_summary(model_dir);
        return Ok(());
    }
    let shard_files: HashSet<&String> = index.weight_map.values().collect();
    for shard_file in shard_files {
        let shard_path = validate_shard_path(model_dir, shard_file)?;
        let loaded = apply(load_shard(&shard_path)?);
        report += assign_loaded_weights(model, loaded);
    }
    eval_loaded_parameters(model)?;
    report.log_summary(model_dir);
    Ok(())
}

/// Assign in-memory, HuggingFace-named weights to a model's parameters and
/// eval them onto the GPU.
///
/// This is the in-memory dual of [`load_generic_weights`]: the safetensors path
/// memory-maps shards from disk, whereas the GGUF inference path dequantizes
/// tensors into a `HashMap` and assigns them here. Behaviour matches the
/// safetensors generic loader exactly — params whose names match are replaced,
/// unmatched map entries are ignored (e.g. `lm_head.weight` for tied models).
pub fn assign_weights<M: ModuleParameters + ModuleParametersExt>(
    model: &mut M,
    weights: HashMap<String, Array>,
) -> Result<(), LoadError> {
    assign_loaded_weights(model, weights);
    eval_loaded_parameters(model)?;
    Ok(())
}

pub(crate) fn load_weights_filtered<F>(
    model_dir: impl AsRef<Path>,
    mut keep_key: F,
) -> Result<HashMap<String, Array>, LoadError>
where
    F: FnMut(&str) -> bool,
{
    let model_dir = model_dir.as_ref();
    let mut all_weights = load_weights_filtered_raw(model_dir, &mut keep_key)?;
    unpack_mlx_quantized_weights(model_dir, &mut all_weights)?;
    dequantize_sidecar_weights(&mut all_weights)?;

    Ok(all_weights
        .into_iter()
        .filter(|(key, _)| keep_key(key) && !is_quant_aux_key(key))
        .collect())
}

/// The tensors `keep_key` wants, plus the quantization aux tensors and sidecars
/// of each, read as stored: nothing is unpacked.
///
/// For a loader that keeps some quantized weights packed; everyone else wants
/// [`load_weights_filtered`], which unpacks.
pub(crate) fn load_weights_filtered_raw<F>(
    model_dir: &Path,
    mut keep_key: F,
) -> Result<HashMap<String, Array>, LoadError>
where
    F: FnMut(&str) -> bool,
{
    let quant_config = load_mlx_quantization_config(model_dir)?;
    // An aux entry has to come along whenever its packed tensor does, or the
    // tensor arrives still packed. Which name that is depends on the layout, so
    // ask about both (see `packed_weight_key_for`).
    let mut wanted = |key: &str| {
        keep_key(key)
            || quant_config.is_some()
                && quant_aux_base_key(key)
                    .is_some_and(|base| keep_key(base) || keep_key(&format!("{base}.weight")))
            || pmetal_runtime::native_loader::quant_sidecar_base(key)
                .is_some_and(|base| keep_key(&format!("{base}.weight")))
    };

    let single_file = model_dir.join("model.safetensors");
    if single_file.exists() {
        return Ok(load_shard(&single_file)?
            .into_iter()
            .filter(|(key, _)| wanted(key))
            .collect());
    }

    let index_path = model_dir.join("model.safetensors.index.json");
    if !index_path.exists() {
        return Err(LoadError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "No weights found",
        )));
    }

    let index_content = std::fs::read_to_string(&index_path).map_err(LoadError::Io)?;
    let index: WeightIndex = serde_json::from_str(&index_content)?;
    let wanted_keys: HashSet<String> = index
        .weight_map
        .keys()
        .filter(|key| wanted(key))
        .cloned()
        .collect();
    let shard_files: HashSet<&String> = index
        .weight_map
        .iter()
        .filter(|(key, _)| wanted_keys.contains(*key))
        .map(|(_, shard_file)| shard_file)
        .collect();

    let mut all_weights = HashMap::new();
    for shard_file in shard_files {
        let shard_path = validate_shard_path(model_dir, shard_file)?;
        let shard_weights = load_shard(&shard_path)?;
        all_weights.extend(
            shard_weights
                .into_iter()
                .filter(|(key, _)| wanted_keys.contains(key)),
        );
    }
    Ok(all_weights)
}

/// Unpack an MLX-quantized checkpoint's `.scales` / `.biases` triples in place,
/// as `config.json`'s quantization block describes them. A no-op for a
/// checkpoint without one.
pub(crate) fn unpack_mlx_quantized_weights(
    model_dir: &Path,
    weights: &mut HashMap<String, Array>,
) -> Result<(), LoadError> {
    if let Some(config) = load_mlx_quantization_config(model_dir)? {
        dequantize_mlx_quantized_weights(weights, &config);
    }
    Ok(())
}

/// Load all safetensor weights from a model directory.
pub fn load_weights(model_dir: impl AsRef<Path>) -> Result<HashMap<String, Array>, LoadError> {
    load_weights_filtered(model_dir, |_| true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn max_abs_diff(a: &Array, b: &Array) -> f32 {
        let n = a.size();
        let f32_dtype = pmetal_runtime::dtype::F32;
        let mut a = a.as_dtype(f32_dtype);
        let mut b = b.as_dtype(f32_dtype);
        let va = a.to_f32_vec(n).expect("read lhs");
        let vb = b.to_f32_vec(n).expect("read rhs");
        va.iter()
            .zip(vb.iter())
            .map(|(p, q)| (p - q).abs())
            .fold(0.0f32, f32::max)
    }

    /// huggingface_hub's shared blob store puts the bytes at
    /// `<cache>/blobs/<xx>/<hash>`, outside the repo directory, behind two
    /// symlinks. Such a shard is inside the cache and must load; one pointing
    /// anywhere else still must not.
    #[test]
    fn shard_paths_follow_the_shared_blob_store_but_no_further() {
        use std::os::unix::fs::symlink;
        let cache = tempdir().unwrap();
        let root = cache.path();
        let snapshot = root.join("models--org--name/snapshots/abc");
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::create_dir_all(root.join("models--org--name/blobs")).unwrap();
        std::fs::create_dir_all(root.join("blobs/9f")).unwrap();
        std::fs::write(root.join("blobs/9f/9fce"), b"shard").unwrap();
        symlink(
            "../../blobs/9f/9fce",
            root.join("models--org--name/blobs/etag"),
        )
        .unwrap();
        symlink("../../blobs/etag", snapshot.join("model-1.safetensors")).unwrap();
        assert!(validate_shard_path(&snapshot, "model-1.safetensors").is_ok());

        let outside = tempdir().unwrap();
        std::fs::write(outside.path().join("elsewhere"), b"shard").unwrap();
        symlink(
            outside.path().join("elsewhere"),
            snapshot.join("model-2.safetensors"),
        )
        .unwrap();
        assert!(validate_shard_path(&snapshot, "model-2.safetensors").is_err());
    }

    /// ⚠️ A ModelOpt checkpoint used to load with its sidecars skipped as
    /// "unmatched" and its packed bytes assigned as dense weights, while
    /// reporting every weight matched. Training and eval both read this path.
    #[test]
    fn modelopt_checkpoints_unpack_to_dense() {
        use pmetal_runtime::QuantizedMode;
        let dir = tempdir().unwrap();
        let (out, in_dim) = (4i32, 64i32);
        let values: Vec<f32> = (0..out * in_dim)
            .map(|i| ((i % 13) as f32 - 6.0) / 6.0)
            .collect();
        let w = Array::from_f32_slice(&values, &[out, in_dim]);
        let (packed, scales) = w.quantize_weights_mode(16, 4, QuantizedMode::Nvfp4);
        let fp8 = w.to_fp8();
        let scalar = |v: f32| Array::from_f32_slice(&[v], &[]);

        let down = "model.layers.0.mlp.down_proj";
        let q = "model.layers.0.self_attn.q_proj";
        let mut weights = HashMap::new();
        weights.insert(
            format!("{down}.weight"),
            packed.view(pmetal_runtime::dtype::U8),
        );
        weights.insert(format!("{down}.weight_scale"), scales.clone());
        weights.insert(format!("{down}.weight_scale_2"), scalar(0.37));
        weights.insert(format!("{down}.input_scale"), scalar(0.2));
        weights.insert(format!("{q}.weight"), fp8.clone());
        weights.insert(format!("{q}.weight_scale"), scalar(0.5));
        weights.insert(format!("{q}.input_scale"), scalar(0.2));
        weights.insert(
            "model.norm.weight".to_string(),
            Array::ones(&[in_dim], pmetal_runtime::dtype::F32),
        );
        write_safetensors(&dir.path().join("model.safetensors"), &weights).unwrap();

        let loaded = load_weights(dir.path()).unwrap();

        let sidecars: Vec<&String> = loaded
            .keys()
            .filter(|k| pmetal_runtime::native_loader::quant_sidecar_base(k).is_some())
            .collect();
        assert!(sidecars.is_empty(), "sidecars survived: {sidecars:?}");

        let want_down = packed
            .dequantize_mode(&scales, None, 16, 4, QuantizedMode::Nvfp4)
            .as_dtype(pmetal_runtime::dtype::F32)
            .multiply(&scalar(0.37));
        let diff = max_abs_diff(&loaded[&format!("{down}.weight")], &want_down);
        assert!(diff < 1e-6, "nvfp4 unpacked wrongly: max|diff| = {diff}");

        let want_q = fp8
            .from_fp8(pmetal_runtime::dtype::F32)
            .multiply(&scalar(0.5));
        let diff = max_abs_diff(&loaded[&format!("{q}.weight")], &want_q);
        assert!(diff < 1e-6, "fp8 unpacked wrongly: max|diff| = {diff}");
    }

    /// A weight scale with nothing to apply it to has to stop the load.
    #[test]
    fn an_orphaned_weight_scale_is_an_error() {
        let dir = tempdir().unwrap();
        let mut weights = HashMap::new();
        weights.insert(
            "model.layers.0.mlp.down_proj.weight_scale_inv".to_string(),
            Array::from_f32_slice(&[1.0], &[1, 1]),
        );
        weights.insert(
            "model.norm.weight".to_string(),
            Array::ones(&[4], pmetal_runtime::dtype::F32),
        );
        write_safetensors(&dir.path().join("model.safetensors"), &weights).unwrap();

        let err = load_weights(dir.path()).unwrap_err().to_string();
        assert!(err.contains("weight_scale_inv"), "unexpected error: {err}");
    }

    fn write_safetensors(
        path: &Path,
        weights: &HashMap<String, Array>,
    ) -> Result<(), pmetal_runtime::compat::IoError> {
        let entries: Vec<_> = weights
            .iter()
            .map(|(key, value)| (key.as_str(), value))
            .collect();
        Array::save_safetensors(path.to_string_lossy().as_ref(), &entries)?;
        Ok(())
    }

    #[test]
    fn load_weights_filtered_reads_only_needed_shards() {
        let temp = tempdir().unwrap();
        let model_dir = temp.path();
        let text_shard = model_dir.join("text.safetensors");
        let broken_shard = model_dir.join("broken.safetensors");

        let mut text_weights = HashMap::new();
        text_weights.insert(
            "model.embed_tokens.weight".to_string(),
            Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]),
        );
        write_safetensors(&text_shard, &text_weights).unwrap();
        std::fs::write(&broken_shard, b"not safetensors").unwrap();

        let index = serde_json::json!({
            "metadata": {},
            "weight_map": {
                "model.embed_tokens.weight": "text.safetensors",
                "model.language_model.layers.0.mlp.experts.gate_up_proj": "broken.safetensors"
            }
        });
        std::fs::write(
            model_dir.join("model.safetensors.index.json"),
            serde_json::to_string_pretty(&index).unwrap(),
        )
        .unwrap();

        let loaded =
            load_weights_filtered(model_dir, |key| !key.contains(".mlp.experts.")).unwrap();

        assert_eq!(loaded.len(), 1);
        assert!(loaded.contains_key("model.embed_tokens.weight"));
    }

    #[test]
    fn load_mlx_quantization_config_accepts_alias_and_overrides() {
        let temp = tempdir().unwrap();
        std::fs::write(
            temp.path().join("config.json"),
            r#"{
                "quantization": {
                    "bits": 4,
                    "group_size": 64,
                    "per_tensor_overrides": {
                        "model.layers.0.self_attn.q_proj.weight": 8
                    }
                }
            }"#,
        )
        .unwrap();

        let config = load_mlx_quantization_config(temp.path())
            .unwrap()
            .expect("quantization config");
        assert_eq!(config.bits, 4);
        assert_eq!(config.group_size, 64);
        assert_eq!(
            config.for_tensor("model.layers.0.self_attn.q_proj.weight"),
            TensorQuant {
                bits: 8,
                group_size: 64
            }
        );
    }

    /// The layout MLX itself writes, and the one mlx-community QAT releases
    /// ship: per-module entries sitting beside `bits` and `group_size`, keyed
    /// by module path rather than tensor key.
    #[test]
    fn load_mlx_quantization_config_reads_mlx_per_module_overrides() {
        let temp = tempdir().unwrap();
        std::fs::write(
            temp.path().join("config.json"),
            r#"{
                "quantization": {
                    "group_size": 64,
                    "bits": 4,
                    "mode": "affine",
                    "language_model.model.layers.0.mlp.gate_proj": {
                        "bits": 8,
                        "group_size": 32
                    },
                    "language_model.model.layers.0.self_attn.q_proj": false
                }
            }"#,
        )
        .unwrap();

        let config = load_mlx_quantization_config(temp.path())
            .unwrap()
            .expect("quantization config");

        assert_eq!(
            config.for_tensor("language_model.model.layers.0.mlp.gate_proj.weight"),
            TensorQuant {
                bits: 8,
                group_size: 32
            },
            "per-module override was not applied, so this tensor unpacks at the \
             model-wide width"
        );
        assert_eq!(
            config.for_tensor("language_model.model.layers.0.mlp.down_proj.weight"),
            TensorQuant {
                bits: 4,
                group_size: 64
            },
            "a module with no override should take the model-wide setting"
        );
        // `false` means the module was left unquantized. It carries no
        // scales/biases, so it never reaches dequantization.
        assert!(
            !config
                .overrides
                .contains_key("language_model.model.layers.0.self_attn.q_proj.weight"),
            "an unquantized module was recorded as an override"
        );
    }

    /// A per-module entry that omits `group_size` inherits the model-wide one,
    /// which is how MLX's `to_quantized(**params)` fills its defaults.
    #[test]
    fn a_partial_per_module_override_inherits_the_model_wide_group_size() {
        let temp = tempdir().unwrap();
        std::fs::write(
            temp.path().join("config.json"),
            r#"{
                "quantization": {
                    "group_size": 64,
                    "bits": 4,
                    "model.layers.3.mlp.up_proj": {"bits": 8}
                }
            }"#,
        )
        .unwrap();

        let config = load_mlx_quantization_config(temp.path())
            .unwrap()
            .expect("quantization config");
        assert_eq!(
            config.for_tensor("model.layers.3.mlp.up_proj.weight"),
            TensorQuant {
                bits: 8,
                group_size: 64
            }
        );
    }

    #[test]
    fn load_weights_dequantizes_mlx_affine_quantized_tensors() {
        let temp = tempdir().unwrap();
        let model_dir = temp.path();
        let weight = Array::from_slice(&[1.0f32; 128], &[2, 64]);
        let (packed, scales, biases) = weight.quantize_weights(64, 4);

        let mut weights = HashMap::new();
        weights.insert("linear.weight".to_string(), packed);
        weights.insert("linear.scales".to_string(), scales);
        weights.insert("linear.biases".to_string(), biases);
        write_safetensors(&model_dir.join("model.safetensors"), &weights).unwrap();
        std::fs::write(
            model_dir.join("config.json"),
            r#"{"quantization": {"bits": 4, "group_size": 64}}"#,
        )
        .unwrap();

        let loaded = load_weights(model_dir).unwrap();
        assert_eq!(loaded.len(), 1);
        let restored = loaded.get("linear.weight").unwrap();
        assert_eq!(restored.shape(), &[2, 64]);
        assert!(!loaded.contains_key("linear.scales"));
        assert!(!loaded.contains_key("linear.biases"));
    }

    /// pmetal's own quantizer writes the aux tensors under the full weight name
    /// rather than beside it, so `quantize_and_save_mlx` output has to keep
    /// loading too.
    #[test]
    fn load_weights_dequantizes_pmetals_own_aux_key_layout() {
        let temp = tempdir().unwrap();
        let model_dir = temp.path();
        let weight = Array::from_slice(&[1.0f32; 128], &[2, 64]);
        let (packed, scales, biases) = weight.quantize_weights(64, 4);

        let mut weights = HashMap::new();
        weights.insert("linear.weight".to_string(), packed);
        weights.insert("linear.weight.scales".to_string(), scales);
        weights.insert("linear.weight.biases".to_string(), biases);
        write_safetensors(&model_dir.join("model.safetensors"), &weights).unwrap();
        std::fs::write(
            model_dir.join("config.json"),
            r#"{"quantization": {"bits": 4, "group_size": 64}}"#,
        )
        .unwrap();

        let loaded = load_weights(model_dir).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded.get("linear.weight").unwrap().shape(), &[2, 64]);
    }

    /// A mixed-precision checkpoint, which is what mlx-community QAT releases
    /// ship: most tensors at the model-wide width, some at another.
    ///
    /// Unpacking the 8-bit tensor at the model-wide 4 bits reads the right
    /// number of bytes and produces the wrong shape, so the load "succeeds" and
    /// the failure surfaces later as a norm complaining about its input size.
    #[test]
    fn load_weights_honours_per_module_quantization_widths() {
        let temp = tempdir().unwrap();
        let model_dir = temp.path();

        let wide = Array::from_slice(&[1.0f32; 128], &[2, 64]);
        let (wide_packed, wide_scales, wide_biases) = wide.quantize_weights(64, 8);
        let narrow = Array::from_slice(&[1.0f32; 128], &[2, 64]);
        let (narrow_packed, narrow_scales, narrow_biases) = narrow.quantize_weights(64, 4);

        let mut weights = HashMap::new();
        weights.insert(
            "model.layers.0.mlp.gate_proj.weight".to_string(),
            wide_packed,
        );
        weights.insert(
            "model.layers.0.mlp.gate_proj.scales".to_string(),
            wide_scales,
        );
        weights.insert(
            "model.layers.0.mlp.gate_proj.biases".to_string(),
            wide_biases,
        );
        weights.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            narrow_packed,
        );
        weights.insert(
            "model.layers.0.self_attn.q_proj.scales".to_string(),
            narrow_scales,
        );
        weights.insert(
            "model.layers.0.self_attn.q_proj.biases".to_string(),
            narrow_biases,
        );
        write_safetensors(&model_dir.join("model.safetensors"), &weights).unwrap();

        std::fs::write(
            model_dir.join("config.json"),
            r#"{
                "quantization": {
                    "group_size": 64,
                    "bits": 4,
                    "mode": "affine",
                    "model.layers.0.mlp.gate_proj": {"bits": 8, "group_size": 64}
                }
            }"#,
        )
        .unwrap();

        let loaded = load_weights(model_dir).unwrap();
        assert_eq!(
            loaded
                .get("model.layers.0.mlp.gate_proj.weight")
                .unwrap()
                .shape(),
            &[2, 64],
            "the 8-bit tensor unpacked at the model-wide 4 bits"
        );
        assert_eq!(
            loaded
                .get("model.layers.0.self_attn.q_proj.weight")
                .unwrap()
                .shape(),
            &[2, 64],
            "the tensor with no override should still take the model-wide width"
        );
    }

    /// A checkpoint key that matches no parameter has to come back in the
    /// report. Silently dropping it is how a model ends up running on random
    /// init while looking like it loaded fine.
    #[test]
    fn unmatched_checkpoint_keys_are_reported() {
        let temp = tempdir().unwrap();
        let model_dir = temp.path();

        let mut weights = HashMap::new();
        weights.insert(
            "model.embed_tokens.weight".to_string(),
            Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]),
        );
        weights.insert(
            "model.not_a_parameter.weight".to_string(),
            Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]),
        );
        weights.insert(
            "layers.0.self_attn.q_proj.weight".to_string(),
            Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]),
        );
        write_safetensors(&model_dir.join("model.safetensors"), &weights).unwrap();

        let mut model = crate::architectures::Qwen3ForCausalLM::new(Default::default()).unwrap();
        let mut report = LoadReport::default();
        report += assign_loaded_weights(
            &mut model,
            load_shard(&model_dir.join("model.safetensors")).unwrap(),
        );

        assert_eq!(
            report.loaded, 1,
            "only embed_tokens matches a real parameter"
        );
        let skipped: HashSet<&str> = report.skipped.iter().map(String::as_str).collect();
        assert_eq!(
            skipped,
            HashSet::from([
                "model.not_a_parameter.weight",
                "layers.0.self_attn.q_proj.weight"
            ]),
            "both unmatched keys must be reported, including the un-prefixed layout"
        );
    }

    /// Build a `flatten_params_mut`-shaped map without a real module, so the
    /// namespace probe can be exercised on layouts no architecture ships.
    fn param_map<'a>(names: &[&str], slots: &'a mut [Array]) -> HashMap<String, &'a mut Array> {
        assert_eq!(names.len(), slots.len());
        names
            .iter()
            .map(|name| (*name).to_string())
            .zip(slots.iter_mut())
            .collect()
    }

    fn checkpoint(keys: &[&str]) -> HashMap<String, Array> {
        keys.iter()
            .map(|key| ((*key).to_string(), Array::from_slice(&[0.0f32], &[1])))
            .collect()
    }

    /// `Qwen/Qwen3-Embedding-8B` publishes the text tower with no namespace at
    /// all, so every key missed and the model answered from random init
    /// (issue #19).
    #[test]
    fn unprefixed_checkpoint_resolves_against_the_model_namespace() {
        let mut slots = [
            Array::from_slice(&[0.0f32], &[1]),
            Array::from_slice(&[0.0f32], &[1]),
            Array::from_slice(&[0.0f32], &[1]),
            Array::from_slice(&[0.0f32], &[1]),
        ];
        let params = param_map(
            &[
                "model.embed_tokens.weight",
                "model.layers.0.self_attn.q_proj.weight",
                "model.norm.weight",
                "lm_head.weight",
            ],
            &mut slots,
        );
        let loaded = checkpoint(&[
            "embed_tokens.weight",
            "layers.0.self_attn.q_proj.weight",
            "norm.weight",
        ]);

        assert_eq!(
            detect_namespace_prefix(&params, &loaded).as_deref(),
            Some("model.")
        );
    }

    /// The checkpoint above plus an `lm_head.weight` that *does* line up. An
    /// early-out on "some key matched" would take the identity mapping here and
    /// drop every layer.
    #[test]
    fn one_already_matching_key_does_not_veto_the_shift() {
        let mut slots = [
            Array::from_slice(&[0.0f32], &[1]),
            Array::from_slice(&[0.0f32], &[1]),
            Array::from_slice(&[0.0f32], &[1]),
            Array::from_slice(&[0.0f32], &[1]),
        ];
        let params = param_map(
            &[
                "model.embed_tokens.weight",
                "model.layers.0.self_attn.q_proj.weight",
                "model.norm.weight",
                "lm_head.weight",
            ],
            &mut slots,
        );
        let loaded = checkpoint(&[
            "embed_tokens.weight",
            "layers.0.self_attn.q_proj.weight",
            "norm.weight",
            "lm_head.weight",
        ]);

        assert_eq!(
            detect_namespace_prefix(&params, &loaded).as_deref(),
            Some("model."),
            "three keys under `model.` outweigh the one that already matched"
        );
    }

    #[test]
    fn a_checkpoint_that_already_lines_up_is_left_alone() {
        let mut slots = [
            Array::from_slice(&[0.0f32], &[1]),
            Array::from_slice(&[0.0f32], &[1]),
        ];
        let params = param_map(
            &["model.embed_tokens.weight", "model.norm.weight"],
            &mut slots,
        );
        let loaded = checkpoint(&["model.embed_tokens.weight", "model.norm.weight"]);

        assert_eq!(detect_namespace_prefix(&params, &loaded), None);
    }

    /// Two towers with identically-named sublayers: prefixing onto either is a
    /// coin flip, so the load has to stay put and report the misses.
    #[test]
    fn an_ambiguous_shift_is_refused() {
        let mut slots = [
            Array::from_slice(&[0.0f32], &[1]),
            Array::from_slice(&[0.0f32], &[1]),
        ];
        let params = param_map(
            &[
                "text_model.layers.0.self_attn.q_proj.weight",
                "vision_model.layers.0.self_attn.q_proj.weight",
            ],
            &mut slots,
        );
        let loaded = checkpoint(&["layers.0.self_attn.q_proj.weight"]);

        assert_eq!(detect_namespace_prefix(&params, &loaded), None);
    }

    #[test]
    fn a_layout_no_prefix_explains_is_left_to_the_report() {
        let mut slots = [Array::from_slice(&[0.0f32], &[1])];
        let params = param_map(&["model.embed_tokens.weight"], &mut slots);
        let loaded = checkpoint(&["bert.encoder.layer.0.attention.self.query.weight"]);

        assert_eq!(detect_namespace_prefix(&params, &loaded), None);
    }

    /// End to end on the real `Qwen3ForCausalLM` parameter tree: before the
    /// shift was detected this loaded nothing at all.
    #[test]
    fn unprefixed_qwen3_checkpoint_loads_into_the_model() {
        let temp = tempdir().unwrap();
        let model_dir = temp.path();

        let mut weights = HashMap::new();
        for key in [
            "embed_tokens.weight",
            "norm.weight",
            "layers.0.self_attn.q_proj.weight",
            "layers.0.self_attn.k_proj.weight",
            "layers.0.mlp.gate_proj.weight",
        ] {
            weights.insert(
                key.to_string(),
                Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]),
            );
        }
        write_safetensors(&model_dir.join("model.safetensors"), &weights).unwrap();

        let mut model = crate::architectures::Qwen3ForCausalLM::new(Default::default()).unwrap();
        let report = assign_loaded_weights(
            &mut model,
            load_shard(&model_dir.join("model.safetensors")).unwrap(),
        );

        assert_eq!(report.loaded, weights.len());
        assert!(report.skipped.is_empty(), "skipped: {:?}", report.skipped);
    }

    /// ⚠️ The `simple_load!` arms — llama, qwen2, mistral, gemma, phi, cohere
    /// and most of the rest — reach the model through `load_generic_weights`,
    /// which read shards straight off disk and never consulted the
    /// quantization config. Every `mlx-community` 4-bit checkpoint therefore
    /// arrived still packed, and the model built on `uint32` payloads.
    ///
    /// Asserted on the *parameter*, not on a returned map, because the gap was
    /// between reading the shard and assigning it.
    #[test]
    fn load_generic_weights_unpacks_a_quantized_checkpoint() {
        let temp = tempdir().unwrap();
        let model_dir = temp.path();

        let mut model = crate::architectures::Qwen3ForCausalLM::new(Default::default()).unwrap();
        let hidden = model.model.embed_tokens.weight.shape()[1];
        let dense = Array::from_slice(&vec![0.5f32; (hidden * 2) as usize], &[2, hidden]);
        let (packed, scales, biases) = dense.quantize_weights(64, 4);

        let mut weights = HashMap::new();
        weights.insert("model.layers.0.self_attn.q_proj.weight".to_string(), packed);
        weights.insert("model.layers.0.self_attn.q_proj.scales".to_string(), scales);
        weights.insert("model.layers.0.self_attn.q_proj.biases".to_string(), biases);
        write_safetensors(&model_dir.join("model.safetensors"), &weights).unwrap();
        std::fs::write(
            model_dir.join("config.json"),
            r#"{"quantization": {"bits": 4, "group_size": 64}}"#,
        )
        .unwrap();

        load_generic_weights(&mut model, model_dir).unwrap();

        assert_eq!(
            model.model.layers[0].self_attn.q_proj.weight.shape(),
            &[2, hidden],
            "q_proj kept the packed uint32 width instead of being dequantized"
        );
    }
}
