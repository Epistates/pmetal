//! Model merging orchestration.
//!
//! This module provides the high-level API for running model merges.
//! It coordinates loading models, applying merge algorithms, and saving results.

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::Array;
use tracing::{info, debug, warn};

use crate::{
    MergeConfig, MergeMethodConfig, MergeParameters, Result, MergeError,
    SafetensorsLoader, TensorLoader, TensorWriter,
    MergeMethod, LinearMerge, SlerpMerge, TiesMerge, DareMerge,
    ModelStockMerge, PassthroughMerge, TaskArithmeticMerge,
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
    info!("Using merge method: {} - {}", method.name(), method.description());

    // Validate configuration
    validate_config(config, &*method)?;

    // Load models lazily
    let loaders = load_models(config)?;
    let base_loader = load_base_model(config)?;

    // Get all tensor names (union across all models)
    let tensor_names = collect_tensor_names(&loaders, &base_loader)?;
    info!("Found {} tensors to merge", tensor_names.len());

    // Determine output path
    let output_path = config.output_path.clone()
        .unwrap_or_else(|| std::path::PathBuf::from("merged_model"));

    // Create output writer
    let mut writer = TensorWriter::new(&output_path)?;

    // Process each tensor
    let total = tensor_names.len();
    for (idx, name) in tensor_names.iter().enumerate() {
        if idx % 100 == 0 || idx == total - 1 {
            info!("Processing tensor {}/{}: {}", idx + 1, total, name);
        }

        let merged = merge_tensor(
            name,
            &loaders,
            base_loader.as_ref(),
            &*method,
            config,
        )?;

        writer.write_tensor(name, &merged)?;
    }

    // Finalize output
    writer.finalize()?;
    info!("Merge complete! Output saved to: {:?}", output_path);

    Ok(output_path)
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
        // TODO: Implement remaining methods
        MergeMethodConfig::Della => {
            warn!("DELLA not yet implemented, falling back to TIES");
            Box::new(TiesMerge::new())
        }
        MergeMethodConfig::DellaLinear => {
            warn!("DELLA-Linear not yet implemented, falling back to linear");
            Box::new(LinearMerge::new())
        }
        MergeMethodConfig::Breadcrumbs => {
            warn!("Breadcrumbs not yet implemented, falling back to TIES");
            Box::new(TiesMerge::new())
        }
        MergeMethodConfig::ModelStock => Box::new(ModelStockMerge::new()),
        MergeMethodConfig::Nearswap => {
            warn!("Nearswap not yet implemented, falling back to SLERP");
            Box::new(SlerpMerge::new())
        }
        MergeMethodConfig::Passthrough => Box::new(PassthroughMerge::new()),
    }
}

/// Validate the merge configuration.
fn validate_config(config: &MergeConfig, method: &dyn MergeMethod) -> Result<()> {
    // Check minimum model count
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
            let model_params = config.models.get(idx)
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
    method.merge(
        &tensors,
        base_tensor.as_ref(),
        &params,
        &config.parameters,
    )
}

/// Verify that all tensor shapes match.
fn verify_shapes(
    name: &str,
    tensors: &[Array],
    base: Option<&Array>,
) -> Result<()> {
    if tensors.is_empty() {
        return Ok(());
    }

    let expected_shape = tensors[0].shape().to_vec();

    for (idx, tensor) in tensors.iter().enumerate().skip(1) {
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
        self.parameters.weight = Some(weight);
        self
    }

    /// Set global density for sparsification.
    pub fn density(mut self, density: f32) -> Self {
        self.parameters.density = Some(density);
        self
    }

    /// Set t parameter for SLERP.
    pub fn t(mut self, t: f32) -> Self {
        self.parameters.t = Some(t);
        self
    }

    /// Set lambda scaling factor.
    pub fn lambda(mut self, lambda: f32) -> Self {
        self.parameters.lambda = Some(lambda);
        self
    }

    /// Build the merge configuration.
    pub fn build(self) -> Result<MergeConfig> {
        let method = self.method.ok_or_else(|| {
            MergeError::InvalidConfig("Merge method is required".to_string())
        })?;

        if self.models.is_empty() {
            return Err(MergeError::NotEnoughModels {
                expected: 1,
                actual: 0,
            });
        }

        let models = self.models
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
        assert_eq!(config.parameters.weight, Some(0.5));
    }

    #[test]
    fn test_builder_requires_method() {
        let builder = MergeBuilder::new()
            .add_model("model1");

        let result = builder.build();
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_requires_models() {
        let builder = MergeBuilder::new()
            .method(MergeMethodConfig::Linear);

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
