//! Configuration types for model merging.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Complete merge configuration, typically loaded from YAML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeConfig {
    /// The merge method to use.
    pub merge_method: MergeMethodConfig,

    /// Models to merge.
    pub models: Vec<ModelConfig>,

    /// Base model for task-vector methods (TIES, DARE, etc.).
    #[serde(default)]
    pub base_model: Option<String>,

    /// Output path for merged model.
    #[serde(default)]
    pub output_path: Option<PathBuf>,

    /// Output dtype (float32, float16, bfloat16).
    #[serde(default = "default_dtype")]
    pub dtype: String,

    /// Global parameters that apply to all models.
    #[serde(default)]
    pub parameters: MergeParameters,

    /// Tokenizer configuration.
    #[serde(default)]
    pub tokenizer: Option<TokenizerConfig>,
}

fn default_dtype() -> String {
    "bfloat16".to_string()
}

/// Configuration for a single model in the merge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model path (local or HuggingFace repo ID).
    pub model: String,

    /// Per-model parameters (override global).
    #[serde(default)]
    pub parameters: MergeParameters,
}

/// Merge method configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MergeMethodConfig {
    /// Simple weighted averaging.
    Linear,

    /// Spherical linear interpolation.
    Slerp,

    /// Task Arithmetic merging (Ilharco et al., 2022).
    /// Uses `MergeParameters.lambda` for global scaling and per-model `weight` fields.
    /// Formula: `W_new = W_base + lambda * sum(w_i * (W_i - W_base))`
    TaskArithmetic,

    /// TIES-Merging (Yadav et al., 2023)
    Ties,

    /// Random pruning with TIES sign consensus.
    DareTies,

    /// Random pruning with linear combination.
    DareLinear,

    /// Adaptive magnitude-based pruning with TIES.
    Della,

    /// Adaptive magnitude-based pruning with linear.
    DellaLinear,

    /// Model breadcrumbs (outlier removal).
    Breadcrumbs,

    /// Geometric interpolation based on task vector similarity.
    ModelStock,

    /// Parameter-wise selective interpolation.
    Nearswap,

    /// No-op passthrough (for frankenmerging).
    Passthrough,
}

/// Parameters for merge operations.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MergeParameters {
    /// Weight for this model in the merge (0.0-1.0).
    #[serde(default)]
    pub weight: Option<f32>,

    /// Density for sparsification methods (0.0-1.0).
    /// Keep this fraction of parameters by magnitude.
    #[serde(default)]
    pub density: Option<f32>,

    /// Interpolation parameter for SLERP (0.0=base, 1.0=other).
    #[serde(default)]
    pub t: Option<f32>,

    /// Scaling factor for task vectors.
    #[serde(default)]
    pub lambda: Option<f32>,

    /// Whether to normalize weights to sum to 1.
    #[serde(default)]
    pub normalize: Option<bool>,

    /// Whether to rescale after DARE pruning.
    #[serde(default)]
    pub rescale: Option<bool>,

    /// Epsilon for DELLA adaptive density.
    #[serde(default)]
    pub epsilon: Option<f32>,

    /// Gamma for breadcrumbs outlier removal.
    #[serde(default)]
    pub gamma: Option<f32>,

    /// Use int8 mask for memory efficiency.
    #[serde(default)]
    pub int8_mask: Option<bool>,
}

impl MergeParameters {
    /// Get weight with default of 1.0.
    pub fn weight(&self) -> f32 {
        self.weight.unwrap_or(1.0)
    }

    /// Get density with default of 1.0 (no sparsification).
    pub fn density(&self) -> f32 {
        self.density.unwrap_or(1.0)
    }

    /// Get t with default of 0.5.
    pub fn t(&self) -> f32 {
        self.t.unwrap_or(0.5)
    }

    /// Get lambda with default of 1.0.
    pub fn lambda(&self) -> f32 {
        self.lambda.unwrap_or(1.0)
    }

    /// Get normalize with default of true.
    pub fn normalize(&self) -> bool {
        self.normalize.unwrap_or(true)
    }

    /// Get rescale with default of true (for DARE).
    pub fn rescale(&self) -> bool {
        self.rescale.unwrap_or(true)
    }

    /// Get epsilon with default of 0.1.
    pub fn epsilon(&self) -> f32 {
        self.epsilon.unwrap_or(0.1)
    }

    /// Get gamma with default of 0.01.
    pub fn gamma(&self) -> f32 {
        self.gamma.unwrap_or(0.01)
    }

    /// Merge with another set of parameters (other overrides self).
    pub fn merge_with(&self, other: &MergeParameters) -> MergeParameters {
        MergeParameters {
            weight: other.weight.or(self.weight),
            density: other.density.or(self.density),
            t: other.t.or(self.t),
            lambda: other.lambda.or(self.lambda),
            normalize: other.normalize.or(self.normalize),
            rescale: other.rescale.or(self.rescale),
            epsilon: other.epsilon.or(self.epsilon),
            gamma: other.gamma.or(self.gamma),
            int8_mask: other.int8_mask.or(self.int8_mask),
        }
    }
}

/// Tokenizer configuration for merged model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfig {
    /// Source for tokenizer: "union", "base", or model path.
    #[serde(default = "default_tokenizer_source")]
    pub source: String,
}

fn default_tokenizer_source() -> String {
    "base".to_string()
}

impl MergeConfig {
    /// Load configuration from a YAML file.
    pub fn from_yaml_file(path: impl AsRef<std::path::Path>) -> crate::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Self::from_yaml(&content)
    }

    /// Parse configuration from a YAML string.
    pub fn from_yaml(yaml: &str) -> crate::Result<Self> {
        Ok(serde_yaml::from_str(yaml)?)
    }

    /// Validate the configuration.
    pub fn validate(&self) -> crate::Result<()> {
        // Check we have at least one model
        if self.models.is_empty() {
            return Err(crate::MergeError::NotEnoughModels {
                expected: 1,
                actual: 0,
            });
        }

        // SLERP requires exactly 2 models
        if matches!(self.merge_method, MergeMethodConfig::Slerp) && self.models.len() != 2 {
            return Err(crate::MergeError::InvalidConfig(
                "SLERP requires exactly 2 models".to_string(),
            ));
        }

        // Task vector methods require base model
        match self.merge_method {
            MergeMethodConfig::Ties
            | MergeMethodConfig::DareTies
            | MergeMethodConfig::DareLinear
            | MergeMethodConfig::Della
            | MergeMethodConfig::DellaLinear
            | MergeMethodConfig::Breadcrumbs
            | MergeMethodConfig::ModelStock => {
                if self.base_model.is_none() {
                    return Err(crate::MergeError::BaseModelRequired {
                        method: format!("{:?}", self.merge_method),
                    });
                }
            }
            _ => {}
        }

        Ok(())
    }
}

impl Default for MergeConfig {
    fn default() -> Self {
        Self {
            merge_method: MergeMethodConfig::Linear,
            models: Vec::new(),
            base_model: None,
            output_path: None,
            dtype: default_dtype(),
            parameters: MergeParameters::default(),
            tokenizer: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_linear_config() {
        let yaml = r#"
merge_method: linear
models:
  - model: model_a
    parameters:
      weight: 0.7
  - model: model_b
    parameters:
      weight: 0.3
dtype: float16
"#;

        let config = MergeConfig::from_yaml(yaml).unwrap();
        assert!(matches!(config.merge_method, MergeMethodConfig::Linear));
        assert_eq!(config.models.len(), 2);
        assert_eq!(config.models[0].parameters.weight(), 0.7);
    }

    #[test]
    fn test_parse_ties_config() {
        let yaml = r#"
merge_method: ties
base_model: base_llama
models:
  - model: finetuned_a
    parameters:
      weight: 1.0
      density: 0.7
  - model: finetuned_b
    parameters:
      weight: 0.5
      density: 0.5
parameters:
  normalize: true
  lambda: 1.0
"#;

        let config = MergeConfig::from_yaml(yaml).unwrap();
        assert!(matches!(config.merge_method, MergeMethodConfig::Ties));
        assert_eq!(config.base_model, Some("base_llama".to_string()));
        assert_eq!(config.models[0].parameters.density(), 0.7);
    }

    #[test]
    fn test_slerp_validation() {
        let config = MergeConfig {
            merge_method: MergeMethodConfig::Slerp,
            models: vec![ModelConfig {
                model: "a".to_string(),
                parameters: Default::default(),
            }],
            ..Default::default()
        };

        assert!(config.validate().is_err());
    }

    #[test]
    fn test_ties_requires_base() {
        let config = MergeConfig {
            merge_method: MergeMethodConfig::Ties,
            models: vec![ModelConfig {
                model: "a".to_string(),
                parameters: Default::default(),
            }],
            base_model: None,
            ..Default::default()
        };

        assert!(config.validate().is_err());
    }
}
