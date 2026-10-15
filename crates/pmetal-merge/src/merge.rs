//! Model merging orchestration.
//!
//! This module provides the high-level API for running model merges.
//! It coordinates loading models, applying merge algorithms, and saving results.
//!
//! A merge streams: tensors are read from memory-mapped checkpoints one name
//! at a time, merged in f32, checked, and appended to the output writer
//! before the next is touched, so memory use is a few tensors and not the
//! models.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use pmetal_runtime::compat::Array;
use regex::Regex;
use tracing::{debug, info, warn};

use crate::moe_align::MoeAlignment;
use crate::{
    BreadcrumbsMerge, ConsensusMerge, DareMerge, DellaMerge, LinearMerge, MergeConfig, MergeError,
    MergeMethod, MergeMethodConfig, MergeParameters, ModelStockMerge, MultiSlerpMerge,
    NearswapMerge, PassthroughMerge, RamMerge, Result, SafetensorsLoader, SlerpMerge,
    TaskArithmeticMerge, TensorLoader, TensorWriter, TiesMerge, VocabMismatch,
};

/// Tensor-name pattern that identifies MoE routed-expert weights.
///
/// Matches the `experts.{idx}.` segment that appears in every supported MoE
/// architecture (DeepSeek, Qwen3MoE, Qwen3Next, GPT-OSS, Llama 4, Granite MoE).
///
/// See [`moe_merge_caveat`] for the limitation this is used to detect.
fn moe_expert_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\.experts\.\d+\.").expect("MoE expert regex compiles"))
}

/// Returns true if any tensor name matches the MoE routed-expert pattern.
///
/// Used by [`run_merge`] to surface the expert-permutation caveat at runtime.
pub fn contains_moe_experts<'a, I>(names: I) -> bool
where
    I: IntoIterator<Item = &'a String>,
{
    let re = moe_expert_re();
    names.into_iter().any(|n| re.is_match(n))
}

/// Canonical documentation text for the MoE expert-permutation caveat.
///
/// **Known limitation:** full-model merging (TIES / DARE / linear / SLERP /
/// task-arithmetic / …) operates on tensor names only. For MoE routed experts
/// (`…experts.{i}.…`) this means expert `i` in checkpoint A is always merged
/// with expert `i` in checkpoint B — even if the training runs specialised
/// those slots to semantically different experts. If the two checkpoints did
/// not share a common base model or share the same expert routing order, the
/// merged expert bank can be incoherent. Adapter / LoRA merging is unaffected
/// because PMetal's LoRA path targets the shared expert only (see
/// `pmetal-lora`).
///
/// Mitigations:
/// * Use `base_model` (TIES / DARE) — task vectors relative to a shared base
///   are far more robust than raw-weight mixing.
/// * Only merge MoE checkpoints that branched from a single pretrained base.
/// * Prefer LoRA/adapter merging (`lora_merge`) for cross-run MoE combination.
pub fn moe_merge_caveat() -> &'static str {
    "MoE routed experts are merged by index; different expert specialisations \
     across checkpoints may produce an incoherent expert bank. See \
     `crate::merge::moe_merge_caveat` docs for mitigations."
}

/// Edits a merge makes to the `config.json` it copies from its source so the
/// config still describes the weights.
#[derive(Debug, Default, Clone)]
struct ConfigEdits {
    /// New row count of the vocabulary tensors, after a vocab mismatch was
    /// reconciled.
    vocab_rows: Option<usize>,
    /// For a layer-sliced merge, the source layer each output layer was taken
    /// from (the first source of its slice).
    layer_map: Option<Vec<usize>>,
}

/// Main entry point for running a model merge.
///
/// Dispatches to the layer-sliced pipeline when the configuration has
/// `slices` and runs a flat merge of `models` otherwise.
///
/// # Arguments
/// * `config` - Merge configuration specifying models, method, and parameters
///
/// # Returns
/// Path to the merged model output directory
pub fn run_merge(config: &MergeConfig) -> Result<PathBuf> {
    if config.is_sliced() {
        return run_merge_sliced(config);
    }
    info!("Starting merge with method: {:?}", config.merge_method);

    // Create the merge method
    let method = create_merge_method(&config.merge_method);
    info!(
        "Using merge method: {} - {}",
        method.name(),
        method.description()
    );

    // Validate configuration
    validate_config(config, &*method)?;

    // Load models lazily
    let loaders = load_models(config)?;
    let base_loader = load_base_model(config)?;

    // Get all tensor names (union across all models)
    let mut tensor_names = collect_tensor_names(&loaders, &base_loader);
    info!("Found {} tensors to merge", tensor_names.len());

    let moe_present = contains_moe_experts(&tensor_names);
    if moe_present && !config.align_moe_experts {
        warn!(
            method = method.name(),
            "MoE routed experts detected in merge set — {}",
            moe_merge_caveat()
        );
    }

    // MoE expert permutation alignment (Phase 4.4). When the user opts in
    // and MoE structure is present, compute the per-(model, layer)
    // permutation table and use it to remap tensor names per-model
    // during the merge. The base model uses the identity permutation; if
    // there's no `base_model`, the first model in the merge set is the
    // alignment reference.
    let alignment = if moe_present && config.align_moe_experts {
        // Build the loader pointer set: `base` first, then `loaders[..]`.
        let mut all_loaders: Vec<&dyn TensorLoader> = Vec::with_capacity(loaders.len() + 1);
        if let Some(b) = base_loader.as_ref() {
            all_loaders.push(b);
        }
        for l in &loaders {
            all_loaders.push(l);
        }
        match crate::moe_align::align_experts(&all_loaders) {
            Ok(a) => {
                info!(
                    "MoE alignment computed: {} layer entries across {} models",
                    a.permutations
                        .iter()
                        .map(|p| p.per_layer.len())
                        .sum::<usize>(),
                    a.permutations.len()
                );
                Some(a)
            }
            Err(e) => {
                warn!(
                    "MoE alignment failed: {}; falling back to name-only merge",
                    e
                );
                None
            }
        }
    } else {
        None
    };

    let sidecar_dir = sidecar_source(config)?;

    // Tied lm_head ↔ embed_tokens: when `config.json` advertises the tie
    // AND both tensors live on disk, drop the alias so we merge once.
    // HF loaders re-tie at `from_pretrained()`.
    if let Some(skip) = sidecar_dir.as_deref().and_then(|src| {
        let names: HashSet<String> = tensor_names.iter().cloned().collect();
        crate::loader::detect_tied_embeddings(src, &names)
    }) {
        let (canonical, alias) = skip;
        info!(
            "tied embeddings detected: keeping {}, skipping alias {}",
            canonical, alias
        );
        tensor_names.retain(|n| n != &alias);
    }

    let output_path = output_dir(config);
    let target_dtype = crate::loader::parse_dtype(&config.dtype)?;
    let mut writer = open_writer(config, &output_path, target_dtype)?;

    let flat = FlatMerge {
        config,
        method: &*method,
        loaders: &loaders,
        base_loader: base_loader.as_ref(),
        alignment: alignment.as_ref(),
    };
    let mut edits = ConfigEdits::default();

    // Process each tensor
    let total = tensor_names.len();
    for (idx, name) in tensor_names.iter().enumerate() {
        if idx % 100 == 0 || idx + 1 == total {
            info!("Processing tensor {}/{}: {}", idx + 1, total, name);
        }
        let (merged, vocab_rows) = flat.merge_tensor(name)?;
        if let Some(rows) = vocab_rows {
            edits.vocab_rows = Some(edits.vocab_rows.map_or(rows, |r| r.max(rows)));
        }
        emit_tensor(name, &merged, config, writer.as_mut())?;
    }

    if config.dry_run {
        info!(
            "Dry-run merge complete: {} tensors verified, no output written",
            total
        );
        return Ok(output_path);
    }

    if let Some(writer) = writer {
        writer.finalize()?;
    }
    finish_output(sidecar_dir.as_deref(), &output_path, target_dtype, &edits)?;

    info!("Merge complete! Output saved to: {:?}", output_path);

    Ok(output_path)
}

/// Where the merged model goes.
fn output_dir(config: &MergeConfig) -> PathBuf {
    config
        .output_path
        .clone()
        .unwrap_or_else(|| PathBuf::from("merged_model"))
}

/// The output writer, or `None` for a dry run.
fn open_writer(
    config: &MergeConfig,
    output_path: &Path,
    dtype: safetensors::Dtype,
) -> Result<Option<TensorWriter>> {
    if config.dry_run {
        return Ok(None);
    }
    let mut writer = TensorWriter::with_dtype(output_path, dtype)?;
    if let Some(limit) = config.max_shard_size {
        let limit = usize::try_from(limit)
            .map_err(|_| MergeError::InvalidConfig(format!("max_shard_size {limit} too large")))?;
        writer = writer.with_max_shard_size(limit);
    }
    Ok(Some(writer))
}

/// Check a merged tensor and append it to the output.
///
/// The sanity sweep turns a NaN/inf into a hard error before it reaches disk,
/// and draining the runtime's error slot turns an op that threw (which leaves
/// a placeholder array behind, not an exception) into one as well.
fn emit_tensor(
    name: &str,
    merged: &Array,
    config: &MergeConfig,
    writer: Option<&mut TensorWriter>,
) -> Result<()> {
    let _ = crate::sanity::check_tensor(name, merged, config.sanity)?;
    pmetal_runtime::check_unobserved_error()?;
    if let Some(writer) = writer {
        writer.write_tensor(name, merged)?;
    }
    Ok(())
}

/// Copy the tokenizer and config sidecars into the output and bring its
/// `config.json` in line with the merged weights.
fn finish_output(
    sidecar_dir: Option<&Path>,
    output_path: &Path,
    target_dtype: safetensors::Dtype,
    edits: &ConfigEdits,
) -> Result<()> {
    match sidecar_dir {
        Some(src) => {
            crate::loader::copy_side_files(src, output_path)?;
            patch_output_config(output_path, target_dtype, edits)?;
        }
        None => {
            warn!("no sidecar source resolvable from config; output dir will lack tokenizer/config")
        }
    }
    Ok(())
}

/// The directory whose tokenizer and config files the output inherits.
///
/// `tokenizer.source` picks it: `base` (the default) is the base model, or
/// the first model without one; `first` is the first model; any other value
/// is a directory. `union` (a merged vocabulary) is not implemented and is
/// rejected, because silently copying one tokenizer next to weights meant for
/// another would produce a model that decodes garbage.
fn sidecar_source(config: &MergeConfig) -> Result<Option<PathBuf>> {
    let first_model = || {
        config.models.first().map(|m| m.model.clone()).or_else(|| {
            config
                .slices
                .as_ref()
                .and_then(|s| s.first())
                .and_then(|s| s.sources.first())
                .map(|s| s.model.clone())
        })
    };
    let chosen = match config.tokenizer.as_ref().map(|t| t.source.as_str()) {
        Some("union") => {
            return Err(MergeError::InvalidConfig(
                "tokenizer.source: union is not supported; use `base`, `first`, or a model \
                 directory, and reconcile differing vocabularies with `vocab_mismatch`"
                    .to_string(),
            ));
        }
        Some("first") => first_model(),
        Some("base") | None => config.base_model.clone().or_else(first_model),
        Some(path) => Some(path.to_string()),
    };
    Ok(chosen.and_then(|candidate| {
        let path = PathBuf::from(&candidate);
        if path.is_dir() {
            Some(path)
        } else if path.is_file() {
            path.parent().map(Path::to_path_buf)
        } else {
            warn!("sidecar source {candidate:?} is not a local directory; skipping it");
            None
        }
    }))
}

/// Write the merge's changes into the output's `config.json`: the weight
/// dtype, a changed vocabulary size, and (for sliced merges) the layer count
/// and per-layer lists. A missing or unparseable config is left alone.
fn patch_output_config(
    output_path: &Path,
    dtype: safetensors::Dtype,
    edits: &ConfigEdits,
) -> Result<()> {
    crate::loader::patch_config_dtype(output_path, dtype)?;
    if edits.vocab_rows.is_none() && edits.layer_map.is_none() {
        return Ok(());
    }
    let path = output_path.join("config.json");
    let Ok(raw) = std::fs::read_to_string(&path) else {
        return Ok(());
    };
    let Ok(mut config) = serde_json::from_str::<serde_json::Value>(&raw) else {
        return Ok(());
    };
    let Some(root) = config.as_object_mut() else {
        return Ok(());
    };
    let apply = |map: &mut serde_json::Map<String, serde_json::Value>| {
        if let Some(rows) = edits.vocab_rows {
            if map.contains_key("vocab_size") {
                map.insert("vocab_size".to_string(), rows.into());
            }
        }
        if let Some(layer_map) = &edits.layer_map {
            patch_layer_fields(map, layer_map);
        }
    };
    // Multimodal configs nest the language model's fields under `text_config`.
    if let Some(nested) = root.get_mut("text_config").and_then(|v| v.as_object_mut()) {
        apply(nested);
    }
    apply(root);
    crate::loader::write_atomic(&path, serde_json::to_string_pretty(&config)?.as_bytes())
}

/// Resize `num_hidden_layers` and remap the per-layer lists
/// (`layer_types`, `layers_block_type`) of one config object.
fn patch_layer_fields(map: &mut serde_json::Map<String, serde_json::Value>, layer_map: &[usize]) {
    let Some(old_layers) = map.get("num_hidden_layers").and_then(|v| v.as_u64()) else {
        return;
    };
    map.insert("num_hidden_layers".to_string(), layer_map.len().into());
    for key in ["layer_types", "layers_block_type"] {
        if let Some(list) = map.get(key).and_then(|v| v.as_array()) {
            if list.len() as u64 == old_layers {
                let remapped: Vec<serde_json::Value> = layer_map
                    .iter()
                    .filter_map(|&src| list.get(src).cloned())
                    .collect();
                map.insert(key.to_string(), remapped.into());
            }
        }
    }
}

/// Create the appropriate merge method from configuration.
fn create_merge_method(method: &MergeMethodConfig) -> Box<dyn MergeMethod> {
    match method {
        MergeMethodConfig::Linear => Box::new(LinearMerge::new()),
        MergeMethodConfig::Slerp => Box::new(SlerpMerge::new()),
        MergeMethodConfig::TaskArithmetic => Box::new(TaskArithmeticMerge::new()),
        MergeMethodConfig::Ties => Box::new(TiesMerge::new()),
        MergeMethodConfig::DareTies => Box::new(DareMerge::with_ties()),
        MergeMethodConfig::DareLinear => Box::new(DareMerge::linear()),
        MergeMethodConfig::Della => Box::new(DellaMerge::new()),
        MergeMethodConfig::DellaLinear => Box::new(DellaMerge::linear()),
        MergeMethodConfig::Breadcrumbs => Box::new(BreadcrumbsMerge::new()),
        MergeMethodConfig::ModelStock => Box::new(ModelStockMerge::new()),
        MergeMethodConfig::Consensus => Box::new(ConsensusMerge::new()),
        MergeMethodConfig::Nearswap => Box::new(NearswapMerge::new()),
        MergeMethodConfig::Ram => Box::new(RamMerge::new()),
        MergeMethodConfig::RamPlus => Box::new(RamMerge::plus()),
        MergeMethodConfig::MultiSlerp => Box::new(MultiSlerpMerge::new()),
        MergeMethodConfig::Passthrough => Box::new(PassthroughMerge::new()),
        MergeMethodConfig::Fisher {
            fisher_paths,
            eps,
            fallback_to_mean,
        } => Box::new(
            crate::methods::FisherMerge::new(fisher_paths.clone())
                .with_eps(*eps)
                .with_fallback_to_mean(*fallback_to_mean),
        ),
        MergeMethodConfig::RegMean {
            gram_paths,
            ridge,
            fallback_to_mean,
        } => Box::new(
            crate::methods::RegMeanMerge::new(gram_paths.clone())
                .with_ridge(*ridge)
                .with_fallback_to_mean(*fallback_to_mean),
        ),
    }
}

/// Validate the merge configuration (flat mode only).
///
/// Slice-mode validation is handled by `MergeConfig::validate()` and
/// `run_merge_sliced()`.
fn validate_config(config: &MergeConfig, method: &dyn MergeMethod) -> Result<()> {
    config.validate()?;

    // Check base model requirement
    if method.requires_base_model() && config.base_model.is_none() {
        return Err(MergeError::BaseModelRequired {
            method: method.name().to_string(),
        });
    }

    Ok(())
}

/// Load all input models.
fn load_models(config: &MergeConfig) -> Result<Vec<SafetensorsLoader>> {
    let mut loaders = Vec::with_capacity(config.models.len());

    for model_config in &config.models {
        let path = Path::new(&model_config.model);
        debug!("Loading model: {:?}", path);
        let loader = SafetensorsLoader::new(path)?;
        loaders.push(loader);
    }

    Ok(loaders)
}

/// Load the base model if specified.
fn load_base_model(config: &MergeConfig) -> Result<Option<SafetensorsLoader>> {
    match &config.base_model {
        Some(base_path) => {
            let path = Path::new(base_path);
            debug!("Loading base model: {:?}", path);
            let loader = SafetensorsLoader::new(path)?;
            Ok(Some(loader))
        }
        None => Ok(None),
    }
}

/// Collect all tensor names across all models, sorted.
fn collect_tensor_names(
    loaders: &[SafetensorsLoader],
    base_loader: &Option<SafetensorsLoader>,
) -> Vec<String> {
    let mut all_names = BTreeSet::new();
    for loader in loaders.iter().chain(base_loader.iter()) {
        all_names.extend(loader.tensor_names());
    }
    all_names.into_iter().collect()
}

/// The inputs of a flat merge, tensor by tensor.
struct FlatMerge<'a> {
    config: &'a MergeConfig,
    method: &'a dyn MergeMethod,
    loaders: &'a [SafetensorsLoader],
    base_loader: Option<&'a SafetensorsLoader>,
    alignment: Option<&'a MoeAlignment>,
}

impl FlatMerge<'_> {
    /// Merge a single tensor across all models. With a MoE alignment,
    /// per-model expert tensor names are remapped via the precomputed
    /// permutation table before loading — model `i`'s `experts.K.…` is
    /// rewritten to `experts.π(K).…`, π being the alignment to the base.
    ///
    /// Returns the merged tensor and, when a vocabulary mismatch was
    /// reconciled, the new row count.
    fn merge_tensor(&self, name: &str) -> Result<(Array, Option<usize>)> {
        // Verify the source dtypes agree across the models that hold this tensor.
        // We do this *before* loading because `load_tensor` upcasts to f32, which
        // would erase the source dtype distinction.
        if !self.config.allow_mixed_dtype {
            verify_source_dtypes(name, self.loaders, self.base_loader)?;
        }

        // The base model (if present) sits at index 0 in the alignment, so
        // `loaders[i]` is alignment index `i + 1` when a base is configured.
        let mut tensors = Vec::with_capacity(self.loaders.len());
        let mut params = Vec::with_capacity(self.loaders.len());
        let base_offset = usize::from(self.base_loader.is_some());

        for (idx, loader) in self.loaders.iter().enumerate() {
            let lookup_name = match self.alignment {
                Some(a) => a.remap_tensor_name(idx + base_offset, name),
                None => name.to_string(),
            };
            if loader.has_tensor(&lookup_name) {
                tensors.push(loader.load_tensor(&lookup_name)?);
                params.push(
                    self.config
                        .models
                        .get(idx)
                        .map(|m| m.parameters.clone())
                        .unwrap_or_default(),
                );
            }
        }

        // A tensor only the base model has is carried over unchanged.
        if tensors.is_empty() {
            return match self.base_loader {
                Some(base) if base.has_tensor(name) => Ok((base.load_tensor(name)?, None)),
                _ => Err(MergeError::TensorNotFound(name.to_string())),
            };
        }

        let mut base_tensor = if self.method.requires_base_model() {
            match self.base_loader {
                Some(base) if base.has_tensor(name) => Some(base.load_tensor(name)?),
                _ => {
                    return Err(MergeError::TensorNotFound(format!(
                        "{name} (held by the merged models but missing from the base model)"
                    )));
                }
            }
        } else {
            None
        };

        let vocab_rows = if is_vocab_tensor(name) {
            align_vocab_rows(
                name,
                &mut tensors,
                &mut base_tensor,
                self.config.vocab_mismatch,
            )?
        } else {
            None
        };

        verify_shapes(name, &tensors, base_tensor.as_ref())?;

        // Run the merge — `merge_named` is the dispatch entry; methods that
        // don't care about the name fall through to `merge` via the default impl.
        let merged = self.method.merge_named(
            name,
            &tensors,
            base_tensor.as_ref(),
            &params,
            &self.config.parameters,
        )?;
        Ok((merged, vocab_rows))
    }
}

/// Verify that the source dtype of `name` agrees across every loader that
/// holds it (and the base loader, if present). Loaders that lack the tensor
/// are skipped silently — that's a different error class handled elsewhere.
///
/// We compare the dtype reported by the safetensors header rather than
/// reading the materialized tensor, so this is cheap (no I/O of the data
/// payload).
fn verify_source_dtypes(
    name: &str,
    loaders: &[SafetensorsLoader],
    base_loader: Option<&SafetensorsLoader>,
) -> Result<()> {
    let mut sources: Vec<(String, safetensors::Dtype)> = Vec::new();
    for (idx, loader) in loaders.iter().enumerate() {
        if loader.has_tensor(name) {
            sources.push((format!("model[{idx}]"), loader.tensor_dtype(name)?));
        }
    }
    if let Some(base) = base_loader {
        if base.has_tensor(name) {
            sources.push(("base".to_string(), base.tensor_dtype(name)?));
        }
    }
    check_same_dtype(name, &sources)
}

/// Error with [`MergeError::DtypeMismatch`] unless every labelled dtype is equal.
fn check_same_dtype(name: &str, sources: &[(String, safetensors::Dtype)]) -> Result<()> {
    let Some((_, first)) = sources.first() else {
        return Ok(());
    };
    if sources.iter().any(|(_, d)| d != first) {
        return Err(MergeError::DtypeMismatch {
            name: name.to_string(),
            dtypes: sources
                .iter()
                .map(|(label, d)| format!("{label}={d:?}"))
                .collect(),
        });
    }
    Ok(())
}

/// Verify that all tensor shapes match.
fn verify_shapes(name: &str, tensors: &[Array], base: Option<&Array>) -> Result<()> {
    if tensors.is_empty() {
        return Ok(());
    }

    let expected_shape = tensors[0].shape().to_vec();

    for tensor in tensors.iter().skip(1) {
        let actual = tensor.shape().to_vec();
        if actual != expected_shape {
            return Err(MergeError::ShapeMismatch {
                name: name.to_string(),
                expected: expected_shape,
                actual,
            });
        }
    }

    if let Some(base_tensor) = base {
        let actual = base_tensor.shape().to_vec();
        if actual != expected_shape {
            return Err(MergeError::ShapeMismatch {
                name: format!("{} (base)", name),
                expected: expected_shape,
                actual,
            });
        }
    }

    Ok(())
}

// =============================================================================
// Vocabulary mismatch
// =============================================================================

/// Whether `name` is a 2-D-by-convention tensor indexed by token id
/// (input embedding or output head).
fn is_vocab_tensor(name: &str) -> bool {
    const VOCAB_SUFFIXES: [&str; 7] = [
        "embed_tokens.weight",
        "lm_head.weight",
        "wte.weight",
        "word_embeddings.weight",
        "embed_out.weight",
        "tok_embeddings.weight",
        "output.weight",
    ];
    VOCAB_SUFFIXES.iter().any(|s| name.ends_with(s))
}

/// The first `rows` rows of a tensor.
fn take_rows(tensor: &Array, rows: i32) -> Array {
    let shape = tensor.shape();
    let starts = vec![0; shape.len()];
    let mut stops = shape.to_vec();
    stops[0] = rows;
    tensor.slice(&starts, &stops)
}

/// Rows `[from, to)` of a tensor.
fn row_range(tensor: &Array, from: i32, to: i32) -> Array {
    let shape = tensor.shape();
    let mut starts = vec![0; shape.len()];
    starts[0] = from;
    let mut stops = shape.to_vec();
    stops[0] = to;
    tensor.slice(&starts, &stops)
}

/// Reconcile vocabulary tensors with different row counts according to
/// `policy`, in place. Returns the common row count when it had to change
/// something.
fn align_vocab_rows(
    name: &str,
    tensors: &mut [Array],
    base: &mut Option<Array>,
    policy: VocabMismatch,
) -> Result<Option<usize>> {
    if tensors.iter().chain(base.iter()).any(|t| t.ndim() < 2) {
        return Ok(None);
    }
    let rows: Vec<i32> = tensors
        .iter()
        .chain(base.iter())
        .map(|t| t.dim(0))
        .collect();
    let (min, max) = (
        *rows.iter().min().expect("at least one tensor"),
        *rows.iter().max().expect("at least one tensor"),
    );
    if min == max {
        return Ok(None);
    }

    match policy {
        VocabMismatch::Error => Err(MergeError::InvalidConfig(format!(
            "tensor {name}: the models have different vocabulary sizes ({rows:?} rows); set \
             `vocab_mismatch: truncate` or `extend` if they share a tokenizer prefix"
        ))),
        VocabMismatch::Truncate => {
            warn!("{name}: truncating {rows:?} rows to {min}");
            for t in tensors.iter_mut().chain(base.iter_mut()) {
                if t.dim(0) > min {
                    *t = take_rows(t, min);
                }
            }
            Ok(Some(min as usize))
        }
        VocabMismatch::Extend => {
            warn!("{name}: extending {rows:?} rows to {max}");
            // The longest model supplies rows nobody else has.
            let longest = tensors
                .iter()
                .max_by_key(|t| t.dim(0))
                .expect("at least one tensor")
                .clone();
            // Missing rows come from the base (itself topped up from the
            // longest model) so they merge as no change, or from the longest
            // model when there is no base.
            let reference = match base.as_ref() {
                Some(b) if b.dim(0) < max => {
                    b.concatenate_2(&row_range(&longest, b.dim(0), max), 0)
                }
                Some(b) => b.clone(),
                None => longest,
            };
            for t in tensors.iter_mut() {
                if t.dim(0) < max {
                    *t = t.concatenate_2(&row_range(&reference, t.dim(0), max), 0);
                }
            }
            if let Some(b) = base.as_mut() {
                *b = reference;
            }
            Ok(Some(max as usize))
        }
    }
}

// =============================================================================
// Slice-based frankenmerging
// =============================================================================

/// Run a frankenmerge using the slice-based configuration.
///
/// Slice-based merging assembles a model by pulling specific layer ranges from
/// one or more source models and optionally merging multiple sources per slice.
///
/// # Layer Remapping
///
/// The layers live under one container whose tensor names look like
/// `<prefix>.<N>.<rest>` (`model.layers.N`, `transformer.h.N`, …); it is
/// detected as the prefix with the most distinct layer indices, so a vision
/// tower's own `layers.N` is not mistaken for it. For each output slice the
/// source layer indices are remapped to a contiguous output range:
///
/// ```text
/// slice[0]: model_a layers [0,16)  →  output layers [0,16)
/// slice[1]: model_b layers [8,24)  →  output layers [16,32)
/// ```
///
/// Tensor `model.layers.8.self_attn.q_proj.weight` from `model_b` becomes
/// `model.layers.16.self_attn.q_proj.weight` in the output. When a slice
/// merges several sources, they are combined layer by layer, and a task-vector
/// method takes its base from the layer the slice's *first* source reads.
///
/// # Non-layer Tensors
///
/// Tensors outside the layer container come from the first source of one
/// slice: the input side (`embed_tokens`, position embeddings, anything not
/// listed below) from the first slice, and the output side (the final norm,
/// `lm_head`) from the last slice, so a model that ends in different layers
/// ends with their norm and head. The output's `config.json` gets the new
/// `num_hidden_layers`, and per-layer lists such as `layer_types` follow the
/// layers they describe.
///
/// # Arguments
/// * `config` - Merge configuration with `slices` populated.
///
/// # Returns
/// Path to the merged model output directory.
pub fn run_merge_sliced(config: &MergeConfig) -> Result<PathBuf> {
    config.validate()?;

    let slices = config.slices.as_ref().ok_or_else(|| {
        MergeError::InvalidConfig("run_merge_sliced called on non-sliced config".to_string())
    })?;

    info!("Starting slice-based frankenmerge: {} slices", slices.len());

    let output_path = output_dir(config);
    let target_dtype = crate::loader::parse_dtype(&config.dtype)?;
    let mut writer = open_writer(config, &output_path, target_dtype)?;
    let sidecar_dir = sidecar_source(config)?;

    let global_base_loader: Option<SafetensorsLoader> = match &config.base_model {
        Some(p) => Some(SafetensorsLoader::new(Path::new(p))?),
        None => None,
    };

    let total_output_layers: usize = slices.iter().map(|s| s.sources[0].n_layers()).sum();
    info!(
        "Output model will have {} transformer layers across {} slices",
        total_output_layers,
        slices.len()
    );

    let mut layer_map: Vec<usize> = Vec::with_capacity(total_output_layers);

    for (slice_idx, slice) in slices.iter().enumerate() {
        let effective_method_cfg = slice.merge_method.as_ref().unwrap_or(&config.merge_method);
        let method = create_merge_method(effective_method_cfg);

        // Effective parameters: global → slice-level (slice wins)
        let effective_params = config.parameters.merge_with(&slice.parameters);

        // Effective base model: global → slice-level override
        let slice_base_loader: Option<SafetensorsLoader> = match &slice.base_model {
            Some(p) => Some(SafetensorsLoader::new(Path::new(p))?),
            None => None,
        };
        let base_loader: Option<&SafetensorsLoader> =
            slice_base_loader.as_ref().or(global_base_loader.as_ref());

        let src_loaders: Vec<SafetensorsLoader> = slice
            .sources
            .iter()
            .map(|src| SafetensorsLoader::new(Path::new(&src.model)))
            .collect::<Result<_>>()?;

        let prefix = layer_container(&src_loaders[0]).ok_or_else(|| {
            MergeError::InvalidConfig(format!(
                "slice[{slice_idx}]: {} has no numbered layer tensors (`<prefix>.<N>.<name>`)",
                slice.sources[0].model
            ))
        })?;
        let out_start = layer_map.len();
        let n_layers = slice.sources[0].n_layers();
        info!(
            "Processing slice[{slice_idx}]: output layers [{out_start}..{})",
            out_start + n_layers
        );

        let first_names = src_loaders[0].tensor_names();
        let mut slice_tensor_count = 0usize;
        for delta in 0..n_layers {
            let out_layer = out_start + delta;
            let first_layer = slice.sources[0].layer_range.0 + delta;
            layer_map.push(first_layer);

            let first_prefix = format!("{prefix}.{first_layer}.");
            let suffixes: Vec<&str> = first_names
                .iter()
                .filter_map(|n| n.strip_prefix(&first_prefix))
                .collect();
            if suffixes.is_empty() {
                return Err(MergeError::InvalidConfig(format!(
                    "slice[{slice_idx}]: {} has no tensors for layer {first_layer} (layer_range \
                     {:?})",
                    slice.sources[0].model, slice.sources[0].layer_range
                )));
            }

            for suffix in suffixes {
                let out_name = format!("{prefix}.{out_layer}.{suffix}");

                // The matching tensor of each source; a source may take a
                // different layer of its model for the same output layer.
                let mut tensors: Vec<Array> = Vec::with_capacity(src_loaders.len());
                let mut per_src_params: Vec<MergeParameters> =
                    Vec::with_capacity(src_loaders.len());
                let mut dtypes: Vec<(String, safetensors::Dtype)> = Vec::new();
                for (src_idx, (loader, src)) in src_loaders.iter().zip(&slice.sources).enumerate() {
                    let src_name = format!("{prefix}.{}.{suffix}", src.layer_range.0 + delta);
                    if loader.has_tensor(&src_name) {
                        dtypes.push((
                            format!("sources[{src_idx}]"),
                            loader.tensor_dtype(&src_name)?,
                        ));
                        tensors.push(loader.load_tensor(&src_name)?);
                        per_src_params.push(effective_params.merge_with(&src.parameters));
                    } else {
                        debug!(
                            "slice[{slice_idx}] src[{src_idx}]: tensor {src_name} not found, skipping"
                        );
                    }
                }

                let base_tensor = if method.requires_base_model() {
                    let base_name = format!("{prefix}.{first_layer}.{suffix}");
                    match base_loader {
                        Some(base) if base.has_tensor(&base_name) => {
                            if !config.allow_mixed_dtype {
                                dtypes.push(("base".to_string(), base.tensor_dtype(&base_name)?));
                            }
                            Some(base.load_tensor(&base_name)?)
                        }
                        _ => {
                            return Err(MergeError::TensorNotFound(format!(
                                "{base_name} (slice[{slice_idx}]'s base model)"
                            )));
                        }
                    }
                } else {
                    None
                };
                if !config.allow_mixed_dtype {
                    check_same_dtype(&out_name, &dtypes)?;
                }

                verify_shapes(&out_name, &tensors, base_tensor.as_ref())?;

                // Route through `merge_named` so name-aware methods
                // (Fisher, RegMean) work in the slice pipeline too.
                let merged = method.merge_named(
                    &out_name,
                    &tensors,
                    base_tensor.as_ref(),
                    &per_src_params,
                    &effective_params,
                )?;
                emit_tensor(&out_name, &merged, config, writer.as_mut())?;
                slice_tensor_count += 1;
            }
        }
        info!("slice[{slice_idx}] complete: {slice_tensor_count} layer tensors written");
    }

    // Tensors outside the layer stack: the input side from the first slice's
    // first source, the output side from the last slice's.
    let first_source = SafetensorsLoader::new(Path::new(&slices[0].sources[0].model))?;
    let last_source =
        SafetensorsLoader::new(Path::new(&slices[slices.len() - 1].sources[0].model))?;
    let mut non_layer: BTreeMap<String, &SafetensorsLoader> = BTreeMap::new();
    for (loader, want_output_side) in [(&first_source, false), (&last_source, true)] {
        let prefix = layer_container(loader);
        for name in loader.tensor_names() {
            let in_stack = prefix.as_deref().is_some_and(|p| is_layer_tensor(p, &name));
            if !in_stack && is_output_side_tensor(&name) == want_output_side {
                non_layer.insert(name, loader);
            }
        }
    }
    if let Some(skip) = sidecar_dir.as_deref().and_then(|src| {
        let names: HashSet<String> = non_layer.keys().cloned().collect();
        crate::loader::detect_tied_embeddings(src, &names)
    }) {
        info!("tied embeddings detected: skipping alias {}", skip.1);
        non_layer.remove(&skip.1);
    }
    for (name, loader) in &non_layer {
        let tensor = loader.load_tensor(name)?;
        emit_tensor(name, &tensor, config, writer.as_mut())?;
        debug!("Copied non-layer tensor: {name}");
    }
    info!("Copied {} non-layer tensors", non_layer.len());

    if config.dry_run {
        info!("Dry-run slice merge complete: no output written");
        return Ok(output_path);
    }
    if let Some(writer) = writer {
        writer.finalize()?;
    }
    finish_output(
        sidecar_dir.as_deref(),
        &output_path,
        target_dtype,
        &ConfigEdits {
            vocab_rows: None,
            layer_map: Some(layer_map),
        },
    )?;

    info!(
        "Slice-based merge complete! Output saved to: {:?}",
        output_path
    );

    Ok(output_path)
}

/// Whether `name` is output-side: the final norm or the head. These come from
/// the last slice of a frankenmerge; everything else outside the layers is
/// input-side.
fn is_output_side_tensor(name: &str) -> bool {
    name.contains("lm_head")
        || name.contains("embed_out")
        || name.contains("ln_f")
        || name.contains("final_layer")
        || name.ends_with("norm.weight")
        || name.ends_with("norm.bias")
        || name.ends_with("output.weight")
}

/// Whether `name` is `<prefix>.<digits>.<something>`.
fn is_layer_tensor(prefix: &str, name: &str) -> bool {
    name.strip_prefix(prefix)
        .and_then(|rest| rest.strip_prefix('.'))
        .and_then(|rest| rest.split_once('.'))
        .is_some_and(|(index, _)| !index.is_empty() && index.bytes().all(|b| b.is_ascii_digit()))
}

/// The prefix of the model's layer stack: of all `<prefix>.<N>.` containers,
/// the one with the most distinct indices (ties go to the shorter prefix).
fn layer_container(loader: &SafetensorsLoader) -> Option<String> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        Regex::new(r"^(.*?(?:layers|h|blocks|layer))\.(\d+)\.").expect("layer regex compiles")
    });
    let mut indices: HashMap<String, BTreeSet<usize>> = HashMap::new();
    for name in loader.tensor_names() {
        if let Some(caps) = re.captures(&name) {
            if let Ok(index) = caps[2].parse() {
                indices
                    .entry(caps[1].to_string())
                    .or_default()
                    .insert(index);
            }
        }
    }
    indices
        .into_iter()
        .max_by(|(pa, a), (pb, b)| a.len().cmp(&b.len()).then(pb.len().cmp(&pa.len())))
        .map(|(prefix, _)| prefix)
}

/// Compile the layer-index regex (matches `model.layers.{N}.` prefix).
fn layer_index_regex() -> Regex {
    // Matches the canonical transformer layer naming convention:
    //   model.layers.<N>.<rest>
    // Also handles models that use `transformer.h.<N>` or similar via a
    // broader pattern that captures any `layers.<N>` or `.N.` between
    // known layer container names.
    Regex::new(r"(?:^|\.)(layers|h)\.(\d+)\.").expect("layer index regex is valid")
}

/// Extract the layer index from a tensor name, if present.
///
/// Returns `None` for non-layer tensors (embed_tokens, lm_head, norm, etc.).
pub fn extract_layer_index(name: &str) -> Option<usize> {
    let re = layer_index_regex();
    re.captures(name)
        .and_then(|caps| caps.get(2))
        .and_then(|m| m.as_str().parse().ok())
}

/// Rewrite the layer index in a tensor name.
///
/// `model.layers.5.self_attn.q_proj.weight` with `new_idx=2` →
/// `model.layers.2.self_attn.q_proj.weight`
///
/// Returns `None` for names that contain no `layers.N` component.
pub fn remap_layer_index(name: &str, new_idx: usize) -> Option<String> {
    let re = layer_index_regex();
    let caps = re.captures(name)?;
    let full_match = caps.get(0)?;
    let container = caps.get(1)?.as_str();
    let match_start = full_match.start();
    let match_end = full_match.end();

    // Determine whether the match was preceded by a '.' separator or is at
    // the start of the string (the regex captures the leading dot when present).
    let sep = if full_match.as_str().starts_with('.') {
        "."
    } else {
        ""
    };
    let after_suffix = &name[match_end..]; // everything after `layers.N.`

    Some(
        format!(
            "{}{}{}{}{}.",
            &name[..match_start],
            sep,
            container,
            '.',
            new_idx
        ) + after_suffix,
    )
}

/// Builder for creating merge configurations programmatically.
#[derive(Debug, Default)]
pub struct MergeBuilder {
    method: Option<MergeMethodConfig>,
    models: Vec<String>,
    base_model: Option<String>,
    output_path: Option<PathBuf>,
    parameters: MergeParameters,
    per_model_params: HashMap<usize, MergeParameters>,
    dtype: Option<String>,
}

impl MergeBuilder {
    /// Create a new merge builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the merge method.
    pub fn method(mut self, method: MergeMethodConfig) -> Self {
        self.method = Some(method);
        self
    }

    /// Add a model to merge.
    pub fn add_model(mut self, path: impl Into<String>) -> Self {
        self.models.push(path.into());
        self
    }

    /// Add a model with specific parameters.
    pub fn add_model_with_params(
        mut self,
        path: impl Into<String>,
        params: MergeParameters,
    ) -> Self {
        let idx = self.models.len();
        self.models.push(path.into());
        self.per_model_params.insert(idx, params);
        self
    }

    /// Set the base model for task arithmetic methods.
    pub fn base_model(mut self, path: impl Into<String>) -> Self {
        self.base_model = Some(path.into());
        self
    }

    /// Set the output path.
    pub fn output(mut self, path: impl Into<PathBuf>) -> Self {
        self.output_path = Some(path.into());
        self
    }

    /// Set global merge weight.
    pub fn weight(mut self, weight: f32) -> Self {
        self.parameters.weight = Some(crate::config::ParameterSetting::Scalar(weight));
        self
    }

    /// Set global density for sparsification.
    pub fn density(mut self, density: f32) -> Self {
        self.parameters.density = Some(crate::config::ParameterSetting::Scalar(density));
        self
    }

    /// Set t parameter for SLERP.
    pub fn t(mut self, t: f32) -> Self {
        self.parameters.t = Some(crate::config::ParameterSetting::Scalar(t));
        self
    }

    /// Set lambda scaling factor.
    pub fn lambda(mut self, lambda: f32) -> Self {
        self.parameters.lambda = Some(crate::config::ParameterSetting::Scalar(lambda));
        self
    }

    /// Set the seed of the stochastic methods (DARE, DELLA).
    pub fn seed(mut self, seed: u64) -> Self {
        self.parameters.seed = Some(seed);
        self
    }

    /// Set output dtype ("float16" / "bfloat16" / "float32"). When unset the
    /// `MergeConfig` default applies (`bfloat16`).
    pub fn dtype(mut self, dtype: impl Into<String>) -> Self {
        self.dtype = Some(dtype.into());
        self
    }

    /// Build the merge configuration.
    pub fn build(self) -> Result<MergeConfig> {
        let method = self
            .method
            .ok_or_else(|| MergeError::InvalidConfig("Merge method is required".to_string()))?;

        if self.models.is_empty() {
            return Err(MergeError::NotEnoughModels {
                expected: 1,
                actual: 0,
            });
        }

        let models = self
            .models
            .into_iter()
            .enumerate()
            .map(|(idx, path)| crate::ModelConfig {
                model: path,
                parameters: self.per_model_params.get(&idx).cloned().unwrap_or_default(),
            })
            .collect();

        // Honor the builder's dtype if set; fall back to the same default the
        // YAML deserializer uses (`bfloat16`) so programmatic and YAML
        // construction agree.
        let defaults = MergeConfig::default();
        let dtype = self.dtype.unwrap_or(defaults.dtype);
        // Validate eagerly so the failure surfaces here rather than at save time.
        let _ = crate::loader::parse_dtype(&dtype)?;

        Ok(MergeConfig {
            merge_method: method,
            models,
            base_model: self.base_model,
            output_path: self.output_path,
            dtype,
            parameters: self.parameters,
            ..defaults
        })
    }

    /// Build and run the merge.
    pub fn run(self) -> Result<PathBuf> {
        let config = self.build()?;
        run_merge(&config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_builder() {
        let builder = MergeBuilder::new()
            .method(MergeMethodConfig::Linear)
            .add_model("model1")
            .add_model("model2")
            .weight(0.5)
            .output("output");

        let config = builder.build().unwrap();
        assert!(matches!(config.merge_method, MergeMethodConfig::Linear));
        assert_eq!(config.models.len(), 2);
        assert!((config.parameters.weight() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_builder_requires_method() {
        let builder = MergeBuilder::new().add_model("model1");

        let result = builder.build();
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_requires_models() {
        let builder = MergeBuilder::new().method(MergeMethodConfig::Linear);

        let result = builder.build();
        assert!(result.is_err());
    }

    #[test]
    fn test_passthrough_merge() {
        let merge = PassthroughMerge::new();
        assert_eq!(merge.name(), "passthrough");
        assert!(!merge.requires_base_model());
    }

    #[test]
    fn layer_helpers_agree_on_the_naming_conventions() {
        assert!(is_layer_tensor(
            "model.layers",
            "model.layers.12.mlp.up_proj.weight"
        ));
        assert!(!is_layer_tensor("model.layers", "model.layers.norm.weight"));
        assert!(!is_layer_tensor(
            "model.layers",
            "model.embed_tokens.weight"
        ));
        assert_eq!(
            extract_layer_index("transformer.h.7.attn.c_attn.weight"),
            Some(7)
        );
        assert_eq!(
            remap_layer_index("model.layers.5.self_attn.q_proj.weight", 2).as_deref(),
            Some("model.layers.2.self_attn.q_proj.weight")
        );
        assert!(is_output_side_tensor("model.norm.weight"));
        assert!(is_output_side_tensor("lm_head.weight"));
        assert!(!is_output_side_tensor("model.embed_tokens.weight"));
    }

    #[test]
    fn layer_fields_follow_the_layers_they_describe() {
        let mut config: serde_json::Map<String, serde_json::Value> = serde_json::from_str(
            r#"{"num_hidden_layers": 3, "layer_types": ["full", "sliding", "sliding"],
                "hidden_size": 8}"#,
        )
        .unwrap();
        // Output layers drawn from source layers 2, 0, 0, 1.
        patch_layer_fields(&mut config, &[2, 0, 0, 1]);
        assert_eq!(config["num_hidden_layers"], 4);
        assert_eq!(
            config["layer_types"],
            serde_json::json!(["sliding", "full", "full", "sliding"])
        );
        assert_eq!(config["hidden_size"], 8);
    }
}
