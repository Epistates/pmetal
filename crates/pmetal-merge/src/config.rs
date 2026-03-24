//! Configuration types for model merging.
//!
//! # Flat merge (original)
//!
//! ```yaml
//! merge_method: linear
//! models:
//!   - model: model_a
//!     parameters:
//!       weight: 0.7
//!   - model: model_b
//!     parameters:
//!       weight: 0.3
//! dtype: float16
//! ```
//!
//! # Slice-based frankenmerge
//!
//! Assemble a model by taking specific layer ranges from different source models.
//! Each `OutputSlice` defines a contiguous range of output layers assembled from
//! one or more source model layer ranges.
//!
//! ```yaml
//! merge_method: passthrough
//! dtype: bfloat16
//! slices:
//!   - sources:
//!       - model: /path/to/model_a
//!         layer_range: [0, 16]
//!   - sources:
//!       - model: /path/to/model_b
//!         layer_range: [16, 32]
//!     merge_method: linear
//!     parameters:
//!       weight:
//!         - value: 0.8
//!           filter: "self_attn"
//!         - value: 0.3
//!           filter: "mlp"
//!         - value: 0.5
//! ```

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Conditional / filter-based parameter settings
// ---------------------------------------------------------------------------

/// A single conditional parameter entry that applies when the tensor name
/// contains the given filter string (or always applies if `filter` is `None`).
///
/// Used as an element in a `ParameterSetting` list to vary merge weights based
/// on which component a tensor belongs to (e.g. attention vs. MLP layers).
///
/// # YAML example
/// ```yaml
/// weight:
///   - value: 0.8
///     filter: "self_attn"
///   - value: 0.3
///     filter: "mlp"
///   - value: 0.5   # default fallback — no filter
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConditionalParam {
    /// The numeric parameter value.
    pub value: f32,

    /// Substring filter on tensor names.  When `None` (or `"*"`), this entry
    /// acts as a catch-all default.  The first matching entry wins.
    #[serde(default)]
    pub filter: Option<String>,
}

/// A parameter value that is either a bare scalar or a list of conditional
/// entries evaluated in order against the tensor name.
///
/// # YAML forms
/// ```yaml
/// weight: 0.5                        # scalar — always resolves to 0.5
/// weight:
///   - value: 0.8
///     filter: "self_attn"
///   - value: 0.5                     # default fallback
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ParameterSetting {
    /// A fixed scalar value.
    Scalar(f32),
    /// An ordered list of conditional entries; the first matching entry wins.
    Conditional(Vec<ConditionalParam>),
}

impl ParameterSetting {
    /// Resolve this setting to a concrete `f32` given a tensor name.
    ///
    /// For `Scalar` variants the tensor name is ignored.
    /// For `Conditional` variants the first entry whose `filter` is a substring
    /// of `tensor_name` (or whose `filter` is `None` / `"*"`) is returned.
    /// Returns `None` if no entry matches.
    pub fn resolve(&self, tensor_name: &str) -> Option<f32> {
        match self {
            ParameterSetting::Scalar(v) => Some(*v),
            ParameterSetting::Conditional(entries) => {
                for entry in entries {
                    let matches = match &entry.filter {
                        None => true,
                        Some(f) if f == "*" => true,
                        Some(f) => tensor_name.contains(f.as_str()),
                    };
                    if matches {
                        return Some(entry.value);
                    }
                }
                None
            }
        }
    }

    /// Resolve with a default fallback value when no conditional entry matches.
    pub fn resolve_or(&self, tensor_name: &str, default: f32) -> f32 {
        self.resolve(tensor_name).unwrap_or(default)
    }
}

impl From<f32> for ParameterSetting {
    fn from(v: f32) -> Self {
        ParameterSetting::Scalar(v)
    }
}

// ---------------------------------------------------------------------------
// Core configuration types
// ---------------------------------------------------------------------------

/// Complete merge configuration, typically loaded from YAML.
///
/// Supports two mutually-exclusive execution modes:
/// - **Flat** (`models` field): all models merged together uniformly.
/// - **Slice** (`slices` field): layer-range based frankenmerging where each
///   output slice specifies which source models and layer ranges to draw from.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeConfig {
    /// The merge method to use (can be overridden per slice).
    pub merge_method: MergeMethodConfig,

    /// Models to merge — used in flat mode (mutually exclusive with `slices`).
    #[serde(default)]
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

    /// Global parameters that apply to all models / slices.
    #[serde(default)]
    pub parameters: MergeParameters,

    /// Tokenizer configuration.
    #[serde(default)]
    pub tokenizer: Option<TokenizerConfig>,

    /// Slice-based configuration for frankenmerging.
    ///
    /// When present, uses the slice execution path instead of flat merging.
    /// Each slice defines a contiguous range of output layers assembled from
    /// one or more source model layer ranges.
    #[serde(default)]
    pub slices: Option<Vec<OutputSlice>>,
}

fn default_dtype() -> String {
    "bfloat16".to_string()
}

/// Configuration for a single model in a flat merge.
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

    /// Reinforced Agent Merging (Hu et al., 2025).
    Ram,

    /// Reinforced Agent Merging Plus with tensor-local adaptive rescaling.
    RamPlus,

    /// Multi-model SLERP via barycentric tangent-space interpolation.
    MultiSlerp,

    /// No-op passthrough (for frankenmerging).
    Passthrough,
}

/// Parameters for merge operations.
///
/// Each numeric field can be either a bare scalar or a list of
/// [`ConditionalParam`] entries that are resolved at merge time based on the
/// tensor name being processed.
///
/// # YAML examples
///
/// Scalar (simple):
/// ```yaml
/// parameters:
///   weight: 0.7
///   density: 0.9
/// ```
///
/// Conditional (per-component):
/// ```yaml
/// parameters:
///   weight:
///     - value: 0.8
///       filter: "self_attn"
///     - value: 0.3
///       filter: "mlp"
///     - value: 0.5   # default fallback
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MergeParameters {
    /// Weight for this model in the merge.  Supports conditional filtering.
    #[serde(default)]
    pub weight: Option<ParameterSetting>,

    /// Density for sparsification methods (0.0–1.0).  Supports conditional filtering.
    #[serde(default)]
    pub density: Option<ParameterSetting>,

    /// Interpolation parameter for SLERP (0.0=base, 1.0=other).  Supports conditional filtering.
    #[serde(default)]
    pub t: Option<ParameterSetting>,

    /// Scaling factor for task vectors.  Supports conditional filtering.
    #[serde(default)]
    pub lambda: Option<ParameterSetting>,

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
    // ------------------------------------------------------------------
    // Tensor-name-aware resolvers (preferred; use these in merge logic)
    // ------------------------------------------------------------------

    /// Resolve `weight` for the given tensor name.  Defaults to `1.0`.
    pub fn resolve_weight(&self, tensor_name: &str) -> f32 {
        self.weight
            .as_ref()
            .and_then(|s| s.resolve(tensor_name))
            .unwrap_or(1.0)
    }

    /// Resolve `density` for the given tensor name.  Defaults to `1.0`.
    pub fn resolve_density(&self, tensor_name: &str) -> f32 {
        self.density
            .as_ref()
            .and_then(|s| s.resolve(tensor_name))
            .unwrap_or(1.0)
    }

    /// Resolve `t` for the given tensor name.  Defaults to `0.5`.
    pub fn resolve_t(&self, tensor_name: &str) -> f32 {
        self.t
            .as_ref()
            .and_then(|s| s.resolve(tensor_name))
            .unwrap_or(0.5)
    }

    /// Resolve `lambda` for the given tensor name.  Defaults to `1.0`.
    pub fn resolve_lambda(&self, tensor_name: &str) -> f32 {
        self.lambda
            .as_ref()
            .and_then(|s| s.resolve(tensor_name))
            .unwrap_or(1.0)
    }

    // ------------------------------------------------------------------
    // Context-free accessors (resolve without tensor name — for callers
    // that have not yet threaded tensor_name through the call stack)
    // ------------------------------------------------------------------

    /// Get weight with default of 1.0 (resolved without tensor-name context).
    pub fn weight(&self) -> f32 {
        self.resolve_weight("")
    }

    /// Get density with default of 1.0 (resolved without tensor-name context).
    pub fn density(&self) -> f32 {
        self.resolve_density("")
    }

    /// Get t with default of 0.5 (resolved without tensor-name context).
    pub fn t(&self) -> f32 {
        self.resolve_t("")
    }

    /// Get lambda with default of 1.0 (resolved without tensor-name context).
    pub fn lambda(&self) -> f32 {
        self.resolve_lambda("")
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
    ///
    /// `ParameterSetting` fields: `other` wins when it is `Some`.
    pub fn merge_with(&self, other: &MergeParameters) -> MergeParameters {
        MergeParameters {
            weight: other.weight.clone().or_else(|| self.weight.clone()),
            density: other.density.clone().or_else(|| self.density.clone()),
            t: other.t.clone().or_else(|| self.t.clone()),
            lambda: other.lambda.clone().or_else(|| self.lambda.clone()),
            normalize: other.normalize.or(self.normalize),
            rescale: other.rescale.or(self.rescale),
            epsilon: other.epsilon.or(self.epsilon),
            gamma: other.gamma.or(self.gamma),
            int8_mask: other.int8_mask.or(self.int8_mask),
        }
    }
}

// ---------------------------------------------------------------------------
// Slice-based frankenmerging types
// ---------------------------------------------------------------------------

/// A single output slice: a contiguous range of output layers assembled from
/// one or more source model layer ranges.
///
/// The output layer indices are determined by the position of this slice in
/// the `MergeConfig.slices` vector and the number of layers contributed by
/// previous slices.
///
/// # YAML example
/// ```yaml
/// slices:
///   - sources:
///       - model: /path/to/base_model
///         layer_range: [0, 16]
///   - sources:
///       - model: /path/to/expert_model
///         layer_range: [16, 32]
///     merge_method: linear
///     parameters:
///       weight:
///         - value: 0.8
///           filter: "self_attn"
///         - value: 0.5
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputSlice {
    /// Source layer ranges that contribute to this output slice.
    ///
    /// When multiple sources are given they are merged using the effective
    /// merge method (slice-level `merge_method` overrides the global one).
    pub sources: Vec<InputSlice>,

    /// Base model for this slice.  Overrides `MergeConfig.base_model` when set.
    #[serde(default)]
    pub base_model: Option<String>,

    /// Merge method for this slice.  Overrides the global `merge_method`.
    #[serde(default)]
    pub merge_method: Option<MergeMethodConfig>,

    /// Parameters for this slice (override global parameters).
    #[serde(default)]
    pub parameters: MergeParameters,
}

/// An input layer range taken from a specific model for use in an
/// [`OutputSlice`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputSlice {
    /// Model path (local directory or HuggingFace repo ID).
    pub model: String,

    /// Half-open layer range `[start, end)` to draw from the source model.
    ///
    /// Tensors for layers `start..end` are loaded from this model and
    /// remapped to the output layer indices determined by the slice's position
    /// in the output sequence.
    pub layer_range: (usize, usize),

    /// Per-source parameters (highest priority in the resolution chain).
    #[serde(default)]
    pub parameters: MergeParameters,
}

impl InputSlice {
    /// Number of layers contributed by this input slice.
    pub fn n_layers(&self) -> usize {
        let (start, end) = self.layer_range;
        end.saturating_sub(start)
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

    /// Returns `true` when the config uses slice-based execution.
    pub fn is_sliced(&self) -> bool {
        self.slices.is_some()
    }

    /// Validate the configuration.
    ///
    /// Checks flat-mode constraints and, when `slices` is present,
    /// validates every slice and its source definitions.
    pub fn validate(&self) -> crate::Result<()> {
        if self.is_sliced() {
            self.validate_sliced()
        } else {
            self.validate_flat()
        }
    }

    fn validate_flat(&self) -> crate::Result<()> {
        // At least one model required
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
            | MergeMethodConfig::ModelStock
            | MergeMethodConfig::Ram
            | MergeMethodConfig::RamPlus => {
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

    fn validate_sliced(&self) -> crate::Result<()> {
        let slices = self.slices.as_ref().expect("is_sliced() was true");

        if slices.is_empty() {
            return Err(crate::MergeError::InvalidConfig(
                "slices array must not be empty".to_string(),
            ));
        }

        for (slice_idx, slice) in slices.iter().enumerate() {
            if slice.sources.is_empty() {
                return Err(crate::MergeError::InvalidConfig(format!(
                    "slice[{}] has no sources",
                    slice_idx
                )));
            }

            for (src_idx, src) in slice.sources.iter().enumerate() {
                let (start, end) = src.layer_range;
                if start >= end {
                    return Err(crate::MergeError::InvalidConfig(format!(
                        "slice[{}].sources[{}]: layer_range [{}, {}) is empty or reversed",
                        slice_idx, src_idx, start, end
                    )));
                }
                if src.model.is_empty() {
                    return Err(crate::MergeError::InvalidConfig(format!(
                        "slice[{}].sources[{}]: model path must not be empty",
                        slice_idx, src_idx
                    )));
                }
            }

            // When a slice uses a task-vector method it needs a base model
            let effective_method = slice.merge_method.as_ref().unwrap_or(&self.merge_method);
            match effective_method {
                MergeMethodConfig::Ties
                | MergeMethodConfig::DareTies
                | MergeMethodConfig::DareLinear
                | MergeMethodConfig::Della
                | MergeMethodConfig::DellaLinear
                | MergeMethodConfig::Breadcrumbs
                | MergeMethodConfig::ModelStock
                | MergeMethodConfig::Ram
                | MergeMethodConfig::RamPlus => {
                    let has_base = slice.base_model.is_some() || self.base_model.is_some();
                    if !has_base {
                        return Err(crate::MergeError::BaseModelRequired {
                            method: format!("{:?}", effective_method),
                        });
                    }
                }
                _ => {}
            }
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
            slices: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Existing flat-mode tests (must continue to pass)
    // -------------------------------------------------------------------------

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

    // -------------------------------------------------------------------------
    // ConditionalParam / ParameterSetting tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_parameter_setting_scalar() {
        let setting = ParameterSetting::Scalar(0.75);
        assert_eq!(
            setting.resolve("model.layers.0.self_attn.q_proj.weight"),
            Some(0.75)
        );
        assert_eq!(
            setting.resolve("model.layers.0.mlp.gate_proj.weight"),
            Some(0.75)
        );
    }

    #[test]
    fn test_parameter_setting_conditional_filter() {
        let setting = ParameterSetting::Conditional(vec![
            ConditionalParam {
                value: 0.8,
                filter: Some("self_attn".to_string()),
            },
            ConditionalParam {
                value: 0.3,
                filter: Some("mlp".to_string()),
            },
            ConditionalParam {
                value: 0.5,
                filter: None,
            },
        ]);

        assert_eq!(
            setting.resolve("model.layers.0.self_attn.q_proj.weight"),
            Some(0.8)
        );
        assert_eq!(
            setting.resolve("model.layers.0.mlp.gate_proj.weight"),
            Some(0.3)
        );
        // fallback (no matching filter → catch-all None entry)
        assert_eq!(setting.resolve("model.embed_tokens.weight"), Some(0.5));
    }

    #[test]
    fn test_parameter_setting_conditional_wildcard() {
        let setting = ParameterSetting::Conditional(vec![ConditionalParam {
            value: 0.6,
            filter: Some("*".to_string()),
        }]);
        assert_eq!(setting.resolve("anything"), Some(0.6));
    }

    #[test]
    fn test_parameter_setting_no_match_returns_none() {
        let setting = ParameterSetting::Conditional(vec![ConditionalParam {
            value: 0.8,
            filter: Some("self_attn".to_string()),
        }]);
        assert_eq!(setting.resolve("model.lm_head.weight"), None);
    }

    #[test]
    fn test_conditional_param_yaml_round_trip() {
        let yaml = r#"
weight:
  - value: 0.8
    filter: "self_attn"
  - value: 0.3
    filter: "mlp"
  - value: 0.5
"#;
        let params: MergeParameters = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(
            params.resolve_weight("model.layers.0.self_attn.q_proj.weight"),
            0.8
        );
        assert_eq!(
            params.resolve_weight("model.layers.0.mlp.down_proj.weight"),
            0.3
        );
        // last entry has no filter → default fallback
        assert_eq!(params.resolve_weight("model.embed_tokens.weight"), 0.5);
    }

    #[test]
    fn test_scalar_weight_yaml() {
        let yaml = "weight: 0.42";
        let params: MergeParameters = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(params.weight(), 0.42);
        assert_eq!(params.resolve_weight("some.tensor.name"), 0.42);
    }

    // -------------------------------------------------------------------------
    // Slice config parsing + validation tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_parse_slice_config() {
        let yaml = r#"
merge_method: passthrough
dtype: bfloat16
slices:
  - sources:
      - model: /path/to/model_a
        layer_range: [0, 16]
  - sources:
      - model: /path/to/model_b
        layer_range: [16, 32]
"#;

        let config = MergeConfig::from_yaml(yaml).unwrap();
        assert!(config.is_sliced());
        let slices = config.slices.as_ref().unwrap();
        assert_eq!(slices.len(), 2);
        assert_eq!(slices[0].sources[0].layer_range, (0, 16));
        assert_eq!(slices[1].sources[0].layer_range, (16, 32));
        assert_eq!(slices[0].sources[0].model, "/path/to/model_a");
    }

    #[test]
    fn test_parse_slice_config_with_per_slice_method() {
        let yaml = r#"
merge_method: passthrough
dtype: bfloat16
slices:
  - sources:
      - model: /path/to/model_a
        layer_range: [0, 16]
  - sources:
      - model: /path/to/model_a
        layer_range: [0, 8]
      - model: /path/to/model_b
        layer_range: [8, 16]
    merge_method: linear
    parameters:
      weight:
        - value: 0.8
          filter: "self_attn"
        - value: 0.3
          filter: "mlp"
        - value: 0.5
"#;

        let config = MergeConfig::from_yaml(yaml).unwrap();
        assert!(config.is_sliced());
        let slices = config.slices.as_ref().unwrap();
        assert_eq!(slices.len(), 2);
        assert!(slices[0].merge_method.is_none());
        assert!(matches!(
            slices[1].merge_method,
            Some(MergeMethodConfig::Linear)
        ));

        let w = slices[1]
            .parameters
            .resolve_weight("model.layers.0.self_attn.q_proj.weight");
        assert_eq!(w, 0.8);
        let w_mlp = slices[1]
            .parameters
            .resolve_weight("model.layers.0.mlp.gate_proj.weight");
        assert_eq!(w_mlp, 0.3);
        let w_default = slices[1]
            .parameters
            .resolve_weight("model.embed_tokens.weight");
        assert_eq!(w_default, 0.5);
    }

    #[test]
    fn test_slice_validate_empty_sources() {
        let config = MergeConfig {
            merge_method: MergeMethodConfig::Passthrough,
            slices: Some(vec![OutputSlice {
                sources: vec![],
                base_model: None,
                merge_method: None,
                parameters: Default::default(),
            }]),
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_slice_validate_reversed_range() {
        let config = MergeConfig {
            merge_method: MergeMethodConfig::Passthrough,
            slices: Some(vec![OutputSlice {
                sources: vec![InputSlice {
                    model: "some_model".to_string(),
                    layer_range: (16, 8), // reversed
                    parameters: Default::default(),
                }],
                base_model: None,
                merge_method: None,
                parameters: Default::default(),
            }]),
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_slice_validate_empty_model_path() {
        let config = MergeConfig {
            merge_method: MergeMethodConfig::Passthrough,
            slices: Some(vec![OutputSlice {
                sources: vec![InputSlice {
                    model: "".to_string(), // empty
                    layer_range: (0, 8),
                    parameters: Default::default(),
                }],
                base_model: None,
                merge_method: None,
                parameters: Default::default(),
            }]),
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_valid_slice_config_passes_validation() {
        let config = MergeConfig {
            merge_method: MergeMethodConfig::Passthrough,
            slices: Some(vec![
                OutputSlice {
                    sources: vec![InputSlice {
                        model: "model_a".to_string(),
                        layer_range: (0, 16),
                        parameters: Default::default(),
                    }],
                    base_model: None,
                    merge_method: None,
                    parameters: Default::default(),
                },
                OutputSlice {
                    sources: vec![InputSlice {
                        model: "model_b".to_string(),
                        layer_range: (16, 32),
                        parameters: Default::default(),
                    }],
                    base_model: None,
                    merge_method: None,
                    parameters: Default::default(),
                },
            ]),
            ..Default::default()
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_slice_task_vector_requires_base() {
        let config = MergeConfig {
            merge_method: MergeMethodConfig::Passthrough,
            base_model: None,
            slices: Some(vec![OutputSlice {
                sources: vec![InputSlice {
                    model: "model_a".to_string(),
                    layer_range: (0, 16),
                    parameters: Default::default(),
                }],
                base_model: None,
                merge_method: Some(MergeMethodConfig::Ties), // requires base
                parameters: Default::default(),
            }]),
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_input_slice_n_layers() {
        let src = InputSlice {
            model: "m".to_string(),
            layer_range: (4, 12),
            parameters: Default::default(),
        };
        assert_eq!(src.n_layers(), 8);
    }

    #[test]
    fn test_merge_parameters_merge_with_conditional() {
        // Scalar global, conditional per-model override
        let global = MergeParameters {
            weight: Some(ParameterSetting::Scalar(0.5)),
            ..Default::default()
        };
        let per_model = MergeParameters {
            weight: Some(ParameterSetting::Conditional(vec![
                ConditionalParam {
                    value: 0.9,
                    filter: Some("attn".to_string()),
                },
                ConditionalParam {
                    value: 0.4,
                    filter: None,
                },
            ])),
            ..Default::default()
        };
        let merged = global.merge_with(&per_model);
        assert_eq!(
            merged.resolve_weight("model.layers.0.self_attn.k_proj.weight"),
            0.9
        );
        assert_eq!(merged.resolve_weight("model.embed_tokens.weight"), 0.4);
    }
}
