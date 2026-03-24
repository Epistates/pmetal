//! Model merging orchestration.
//!
//! This module provides the high-level API for running model merges.
//! It coordinates loading models, applying merge algorithms, and saving results.

use std::collections::HashMap;

use mlx_rs::Array;
use regex::Regex;
use tracing::{debug, info};

use crate::{
    BreadcrumbsMerge, DareMerge, DellaMerge, LinearMerge, MergeConfig, MergeError, MergeMethod,
    MergeMethodConfig, MergeParameters, ModelStockMerge, MultiSlerpMerge, NearswapMerge,
    PassthroughMerge, RamMerge, Result, SafetensorsLoader, SlerpMerge, TaskArithmeticMerge,
    TensorLoader, TensorWriter, TiesMerge,
    batched::{BatchConfig, BatchedMerger, MergeStats, StreamingBatchedMerger},
};

/// Main entry point for running a model merge.
///
/// # Arguments
/// * `config` - Merge configuration specifying models, method, and parameters
///
/// # Returns
/// Path to the merged model output directory
pub fn run_merge(config: &MergeConfig) -> Result<std::path::PathBuf> {
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
    let tensor_names = collect_tensor_names(&loaders, &base_loader)?;
    info!("Found {} tensors to merge", tensor_names.len());

    // Determine output path
    let output_path = config
        .output_path
        .clone()
        .unwrap_or_else(|| std::path::PathBuf::from("merged_model"));

    // Create output writer
    let mut writer = TensorWriter::new(&output_path)?;

    // Process each tensor
    let total = tensor_names.len();
    for (idx, name) in tensor_names.iter().enumerate() {
        if idx % 100 == 0 || idx == total - 1 {
            info!("Processing tensor {}/{}: {}", idx + 1, total, name);
        }

        let merged = merge_tensor(name, &loaders, base_loader.as_ref(), &*method, config)?;

        writer.write_tensor(name, &merged)?;
    }

    // Finalize output
    writer.finalize()?;
    info!("Merge complete! Output saved to: {:?}", output_path);

    Ok(output_path)
}

/// Run a model merge using batched processing for improved throughput.
///
/// This variant uses the optimized batched processing pipeline which:
/// - Processes multiple tensors per GPU sync (reduces sync overhead)
/// - Uses O(n) online thresholding instead of O(n log n) sorting
/// - Writes merged tensors immediately (memory-efficient streaming)
///
/// # Arguments
/// * `config` - Merge configuration specifying models, method, and parameters
/// * `batch_config` - Optional batch processing configuration
///
/// # Returns
/// Tuple of (output path, merge statistics)
pub fn run_merge_batched(
    config: &MergeConfig,
    batch_config: Option<BatchConfig>,
) -> Result<(std::path::PathBuf, MergeStats)> {
    let batch_config = batch_config.unwrap_or_default();

    info!(
        "Starting batched merge with method: {:?} (batch_size={})",
        config.merge_method, batch_config.batch_size
    );

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
    let tensor_names = collect_tensor_names(&loaders, &base_loader)?;
    info!("Found {} tensors to merge", tensor_names.len());

    // Determine output path
    let output_path = config
        .output_path
        .clone()
        .unwrap_or_else(|| std::path::PathBuf::from("merged_model"));

    // Create output writer
    let writer = TensorWriter::new(&output_path)?;

    // Create batched merger
    let merger = BatchedMerger::new(
        batch_config,
        &*method,
        &loaders,
        base_loader.as_ref(),
        config,
    );

    // Create streaming merger and process
    let mut streaming = StreamingBatchedMerger::new(merger, writer);
    let stats = streaming.process_all(&tensor_names)?;

    info!(
        "Batched merge complete! {} tensors in {:.1}ms ({:.1} tensors/sec)",
        stats.total_tensors, stats.elapsed_ms, stats.tensors_per_second
    );
    info!("Output saved to: {:?}", output_path);

    Ok((output_path, stats))
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
        MergeMethodConfig::Nearswap => Box::new(NearswapMerge::new()),
        MergeMethodConfig::Ram => Box::new(RamMerge::new()),
        MergeMethodConfig::RamPlus => Box::new(RamMerge::plus()),
        MergeMethodConfig::MultiSlerp => Box::new(MultiSlerpMerge::new()),
        MergeMethodConfig::Passthrough => Box::new(PassthroughMerge::new()),
    }
}

/// Validate the merge configuration (flat mode only).
///
/// Slice-mode validation is handled by `MergeConfig::validate()` and
/// `run_merge_sliced()`.
fn validate_config(config: &MergeConfig, method: &dyn MergeMethod) -> Result<()> {
    // Slice-mode: delegate to config-level validator which covers slices.
    if config.is_sliced() {
        return config.validate();
    }

    // Flat mode: check minimum model count
    if config.models.is_empty() {
        return Err(MergeError::NotEnoughModels {
            expected: 1,
            actual: 0,
        });
    }

    // SLERP requires exactly 2 models
    if matches!(config.merge_method, MergeMethodConfig::Slerp) && config.models.len() != 2 {
        return Err(MergeError::InvalidConfig(format!(
            "SLERP requires exactly 2 models, got {}",
            config.models.len()
        )));
    }

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
        let path = std::path::Path::new(&model_config.model);
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
            let path = std::path::Path::new(base_path);
            debug!("Loading base model: {:?}", path);
            let loader = SafetensorsLoader::new(path)?;
            Ok(Some(loader))
        }
        None => Ok(None),
    }
}

/// Collect all tensor names across all models.
fn collect_tensor_names(
    loaders: &[SafetensorsLoader],
    base_loader: &Option<SafetensorsLoader>,
) -> Result<Vec<String>> {
    let mut all_names = std::collections::HashSet::new();

    for loader in loaders {
        for name in loader.tensor_names() {
            all_names.insert(name);
        }
    }

    if let Some(base) = base_loader {
        for name in base.tensor_names() {
            all_names.insert(name);
        }
    }

    let mut names: Vec<String> = all_names.into_iter().collect();
    names.sort();
    Ok(names)
}

/// Merge a single tensor across all models.
fn merge_tensor(
    name: &str,
    loaders: &[SafetensorsLoader],
    base_loader: Option<&SafetensorsLoader>,
    method: &dyn MergeMethod,
    config: &MergeConfig,
) -> Result<Array> {
    // Load tensors from each model
    let mut tensors = Vec::with_capacity(loaders.len());
    let mut params = Vec::with_capacity(loaders.len());

    for (idx, loader) in loaders.iter().enumerate() {
        if loader.tensor_names().contains(&name.to_string()) {
            let tensor = loader.load_tensor(name)?;
            tensors.push(tensor);

            // Get per-model parameters
            let model_params = config
                .models
                .get(idx)
                .map(|m| m.parameters.clone())
                .unwrap_or_default();
            params.push(model_params);
        }
    }

    // If no tensors found, check base model
    if tensors.is_empty() {
        if let Some(base) = base_loader {
            if base.tensor_names().contains(&name.to_string()) {
                return base.load_tensor(name);
            }
        }
        return Err(MergeError::TensorNotFound(name.to_string()));
    }

    // Load base tensor if needed
    let base_tensor = if method.requires_base_model() {
        if let Some(base) = base_loader {
            if base.tensor_names().contains(&name.to_string()) {
                Some(base.load_tensor(name)?)
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    // Verify shapes match
    verify_shapes(name, &tensors, base_tensor.as_ref())?;

    // Run the merge
    method.merge(&tensors, base_tensor.as_ref(), &params, &config.parameters)
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
// Slice-based frankenmerging
// =============================================================================

/// Run a frankenmerge using the slice-based configuration.
///
/// Slice-based merging assembles a model by pulling specific layer ranges from
/// one or more source models and optionally merging multiple sources per slice.
///
/// # Layer Remapping
///
/// Tensor names follow the convention `model.layers.{N}.{rest}`.  For each
/// output slice the source layer indices are remapped to a contiguous output
/// range.  For example:
///
/// ```text
/// slice[0]: model_a layers [0,16)  →  output layers [0,16)
/// slice[1]: model_b layers [8,24)  →  output layers [16,32)
/// ```
///
/// Tensor `model.layers.8.self_attn.q_proj.weight` from `model_b` becomes
/// `model.layers.16.self_attn.q_proj.weight` in the output.
///
/// # Non-layer Tensors
///
/// Tensors that do not contain a `layers.N` component (e.g. `model.embed_tokens.weight`,
/// `model.norm.weight`, `lm_head.weight`) are copied from the first source model
/// of the first slice unless overridden by `base_model`.
///
/// # Arguments
/// * `config` - Merge configuration with `slices` populated.
///
/// # Returns
/// Path to the merged model output directory.
pub fn run_merge_sliced(config: &MergeConfig) -> Result<std::path::PathBuf> {
    config.validate()?;

    let slices = config.slices.as_ref().ok_or_else(|| {
        MergeError::InvalidConfig("run_merge_sliced called on non-sliced config".to_string())
    })?;

    info!("Starting slice-based frankenmerge: {} slices", slices.len());

    // Pre-compile the layer-index regex once
    let layer_re = layer_index_regex();

    // Determine output path
    let output_path = config
        .output_path
        .clone()
        .unwrap_or_else(|| std::path::PathBuf::from("merged_model"));

    let mut writer = TensorWriter::new(&output_path)?;

    // Track which non-layer tensors we've already written so we don't duplicate
    let mut written_non_layer: std::collections::HashSet<String> = std::collections::HashSet::new();

    // Collect base model loader (optional)
    let global_base_loader: Option<SafetensorsLoader> = match &config.base_model {
        Some(p) => Some(SafetensorsLoader::new(std::path::Path::new(p))?),
        None => None,
    };

    // Compute output layer offsets: output_offset[i] = sum of n_layers for slices [0..i)
    let output_offsets: Vec<usize> = {
        let mut offsets = Vec::with_capacity(slices.len());
        let mut running = 0usize;
        for slice in slices {
            offsets.push(running);
            // Use the layer count of the first source as authoritative
            if let Some(first_src) = slice.sources.first() {
                running += first_src.n_layers();
            }
        }
        offsets
    };

    let total_output_layers: usize = slices
        .iter()
        .filter_map(|s| s.sources.first())
        .map(|src| src.n_layers())
        .sum();

    info!(
        "Output model will have {} transformer layers across {} slices",
        total_output_layers,
        slices.len()
    );

    for (slice_idx, (slice, &out_offset)) in slices.iter().zip(output_offsets.iter()).enumerate() {
        info!(
            "Processing slice[{}]: output layers [{}..{})",
            slice_idx,
            out_offset,
            out_offset + slice.sources.first().map(|s| s.n_layers()).unwrap_or(0)
        );

        // Effective merge method for this slice
        let effective_method_cfg = slice.merge_method.as_ref().unwrap_or(&config.merge_method);
        let method = create_merge_method(effective_method_cfg);

        // Effective parameters: global → slice-level (slice wins)
        let effective_params = config.parameters.merge_with(&slice.parameters);

        // Effective base model: global → slice-level override
        let slice_base_loader: Option<SafetensorsLoader> = match &slice.base_model {
            Some(p) => Some(SafetensorsLoader::new(std::path::Path::new(p))?),
            None => None,
        };
        let base_loader_ref: Option<&SafetensorsLoader> =
            slice_base_loader.as_ref().or(global_base_loader.as_ref());

        // Load source model loaders for this slice
        let src_loaders: Vec<SafetensorsLoader> = slice
            .sources
            .iter()
            .map(|src| SafetensorsLoader::new(std::path::Path::new(&src.model)))
            .collect::<Result<_>>()?;

        // Build the set of (output_name, source_name_per_model, per_source_params)
        // by iterating over layer indices in the source range.
        let (out_src_start, out_src_end) = slice
            .sources
            .first()
            .map(|s| s.layer_range)
            .unwrap_or((0, 0));

        let n_layers = out_src_end.saturating_sub(out_src_start);

        // Collect all output layer tensor names that will be produced by this slice
        // We iterate over the source layers and remap to output layer indices.
        let mut slice_tensor_count = 0usize;

        for layer_delta in 0..n_layers {
            let out_layer_idx = out_offset + layer_delta;

            // Collect all tensor suffixes for this layer from the first source model
            // (suffixes should be the same across sources)
            let src_layer_idx_0 = slice.sources[0].layer_range.0 + layer_delta;
            let layer_prefix_src = format!("model.layers.{}.", src_layer_idx_0);
            let layer_prefix_out = format!("model.layers.{}.", out_layer_idx);

            // Gather all tensor suffixes available for this layer from the first source
            let suffixes: Vec<String> = src_loaders[0]
                .tensor_names()
                .into_iter()
                .filter(|n| n.starts_with(&layer_prefix_src))
                .map(|n| n[layer_prefix_src.len()..].to_string())
                .collect();

            if suffixes.is_empty() {
                debug!(
                    "slice[{}]: no tensors found for source layer {} in model {:?}",
                    slice_idx, src_layer_idx_0, slice.sources[0].model
                );
            }

            for suffix in &suffixes {
                let out_name = format!("{}{}", layer_prefix_out, suffix);

                // Collect the corresponding tensor from each source model
                // (each source may contribute a different layer to the same output layer)
                let mut tensors: Vec<Array> = Vec::with_capacity(src_loaders.len());
                let mut per_src_params: Vec<MergeParameters> =
                    Vec::with_capacity(src_loaders.len());

                for (src_idx, (src_loader, src_def)) in
                    src_loaders.iter().zip(slice.sources.iter()).enumerate()
                {
                    let src_layer_idx = src_def.layer_range.0 + layer_delta;
                    let src_name = format!("model.layers.{}.{}", src_layer_idx, suffix);

                    if src_loader.tensor_names().contains(&src_name) {
                        let tensor = src_loader.load_tensor(&src_name)?;
                        tensors.push(tensor);
                        // Per-source params override slice params
                        let p = effective_params.merge_with(&src_def.parameters);
                        per_src_params.push(p);
                    } else {
                        debug!(
                            "slice[{}] src[{}]: tensor {} not found, skipping",
                            slice_idx, src_idx, src_name
                        );
                    }
                }

                if tensors.is_empty() {
                    debug!(
                        "slice[{}]: no source tensors for {}, skipping",
                        slice_idx, out_name
                    );
                    continue;
                }

                // Load base tensor if the method requires it
                let base_tensor = if method.requires_base_model() {
                    load_base_tensor_for_slice(
                        base_loader_ref,
                        &out_name,
                        &layer_re,
                        out_layer_idx,
                    )?
                } else {
                    None
                };

                verify_shapes(&out_name, &tensors, base_tensor.as_ref())?;

                let merged = method.merge(
                    &tensors,
                    base_tensor.as_ref(),
                    &per_src_params,
                    &effective_params,
                )?;
                writer.write_tensor(&out_name, &merged)?;
                slice_tensor_count += 1;
            }
        }

        // On the first slice, also copy non-layer tensors (embed_tokens, lm_head, etc.)
        // from the first source model.  Only do this once.
        if slice_idx == 0 {
            let first_loader = &src_loaders[0];
            let non_layer_tensors: Vec<String> = first_loader
                .tensor_names()
                .into_iter()
                .filter(|n| !is_layer_tensor(n, &layer_re))
                .collect();

            for name in &non_layer_tensors {
                if !written_non_layer.contains(name) {
                    let tensor = first_loader.load_tensor(name)?;
                    writer.write_tensor(name, &tensor)?;
                    written_non_layer.insert(name.clone());
                    debug!("Copied non-layer tensor: {}", name);
                }
            }

            info!(
                "Copied {} non-layer tensors from first source model",
                non_layer_tensors.len()
            );
        }

        info!(
            "slice[{}] complete: {} layer tensors written",
            slice_idx, slice_tensor_count
        );
    }

    writer.finalize()?;
    info!(
        "Slice-based merge complete! Output saved to: {:?}",
        output_path
    );

    Ok(output_path)
}

/// Returns `true` if the tensor name contains a `layers.N` component.
fn is_layer_tensor(name: &str, re: &Regex) -> bool {
    re.is_match(name)
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

/// Load a base tensor for a slice merge, handling the layer-name remapping.
///
/// The base model stores tensors at their *original* layer indices.  The
/// `out_name` uses the *output* layer index.  We remap back to look up the
/// corresponding tensor in the base model.
fn load_base_tensor_for_slice(
    base_loader: Option<&SafetensorsLoader>,
    out_name: &str,
    layer_re: &Regex,
    _out_layer_idx: usize,
) -> Result<Option<Array>> {
    let Some(base) = base_loader else {
        return Ok(None);
    };

    // For base tensors we use the output name directly — the caller passes the
    // output-remapped name, and we look for it as-is.  If not found, fall back
    // to None (the merge method decides how to handle a missing base tensor).
    if base.tensor_names().contains(&out_name.to_string()) {
        return Ok(Some(base.load_tensor(out_name)?));
    }

    // Not in the base model — acceptable for sliced frankenmerges where the
    // base only covers some layers.
    let _ = layer_re; // suppress unused warning
    Ok(None)
}

/// Builder for creating merge configurations programmatically.
#[derive(Debug, Default)]
pub struct MergeBuilder {
    method: Option<MergeMethodConfig>,
    models: Vec<String>,
    base_model: Option<String>,
    output_path: Option<std::path::PathBuf>,
    parameters: MergeParameters,
    per_model_params: HashMap<usize, MergeParameters>,
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
    pub fn output(mut self, path: impl Into<std::path::PathBuf>) -> Self {
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

        Ok(MergeConfig {
            merge_method: method,
            models,
            base_model: self.base_model,
            output_path: self.output_path,
            dtype: "float16".to_string(),
            parameters: self.parameters,
            tokenizer: None,
            slices: None,
        })
    }

    /// Build and run the merge.
    pub fn run(self) -> Result<std::path::PathBuf> {
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
}
