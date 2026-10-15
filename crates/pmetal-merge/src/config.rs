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

    /// Allow merging tensors whose source dtypes disagree across models.
    /// When `false` (default), a per-tensor dtype mismatch surfaces as
    /// [`crate::MergeError::DtypeMismatch`]. When `true`, all sources are
    /// upcast to f32 in-memory and merging proceeds.
    #[serde(default)]
    pub allow_mixed_dtype: bool,

    /// How aggressively to validate each merged tensor before writing it.
    /// Default is `Quick` (NaN/inf detection only — adds essentially no
    /// cost on top of the f32 materialization the writer performs anyway).
    #[serde(default)]
    pub sanity: crate::sanity::SanityLevel,

    /// When `true`, run the merge pipeline through `verify_shapes` /
    /// `verify_source_dtypes` / sanity checks but **skip** writing any
    /// tensors. Useful for catching shape or dtype mismatches and
    /// enumerating outputs before committing to a long write. Sidecar
    /// files are also not copied.
    #[serde(default)]
    pub dry_run: bool,

    /// Run MoE expert permutation alignment as a pre-pass. When enabled,
    /// each non-base model's expert ordering is solved against the base
    /// model via Hungarian assignment on flattened expert fingerprints,
    /// and tensor names are remapped accordingly during the per-tensor
    /// merge. Applies only when MoE structure is detected. Off by
    /// default to avoid changing behavior of existing flows that target
    /// non-MoE merges.
    #[serde(default)]
    pub align_moe_experts: bool,

    /// What to do when the vocabulary tensors (`embed_tokens`, `lm_head`)
    /// have different row counts across the models. See [`VocabMismatch`].
    #[serde(default)]
    pub vocab_mismatch: VocabMismatch,

    /// Largest output shard in bytes. A tensor larger than this gets a shard
    /// to itself. Defaults to 5 GiB.
    #[serde(default)]
    pub max_shard_size: Option<u64>,
}

/// How to reconcile vocabulary tensors whose row counts differ.
///
/// Fine-tunes often pad or extend the vocabulary of their base (added
/// special tokens, a vocab padded to a multiple of 64). Rows are matched by
/// token id, which is only meaningful when the models share a tokenizer
/// prefix, so the default is to refuse and let the user opt in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum VocabMismatch {
    /// Fail with an error naming the tensor and the shapes. Default.
    #[default]
    Error,
    /// Cut every model to the smallest row count.
    Truncate,
    /// Grow every model to the largest row count. A model's missing rows are
    /// filled from the base model, or from the model that has them when
    /// there is no base (or the base is short too), so they merge as no change.
    Extend,
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

    /// Fisher-weighted averaging (Matena & Raffel, 2022). Each model
    /// contributes its parameters proportional to a precomputed diagonal
    /// Fisher information tensor stored alongside its weights as
    /// `fisher.safetensors`.
    Fisher {
        /// Per-model paths to a `fisher.safetensors` directory. Length
        /// must match the number of models being merged.
        #[serde(default)]
        fisher_paths: Vec<PathBuf>,
        /// Numerical-stability ridge added to the denominator.
        #[serde(default = "default_fisher_eps")]
        eps: f32,
        /// When `true`, tensors absent from a model's Fisher file fall
        /// back to a plain weighted mean instead of erroring out.
        #[serde(default = "default_fisher_fallback")]
        fallback_to_mean: bool,
    },

    /// RegMean closed-form merge (Jin et al., 2023). 2-D weight matrices
    /// are merged via `(Σ G_i)⁻¹ Σ G_i W_i` where each `G_i` is a
    /// per-model Gram matrix `XᵀX` precomputed on a calibration batch.
    RegMean {
        /// Per-model paths to a `regmean_grams.safetensors` directory.
        #[serde(default)]
        gram_paths: Vec<PathBuf>,
        /// Tikhonov ridge for `(Σ G + ridge·I)`.
        #[serde(default = "default_regmean_ridge")]
        ridge: f32,
        /// Mean fallback for tensors without a Gram entry.
        #[serde(default = "default_regmean_fallback")]
        fallback_to_mean: bool,
    },
}

fn default_fisher_eps() -> f32 {
    1e-8
}
fn default_fisher_fallback() -> bool {
    true
}
fn default_regmean_ridge() -> f32 {
    1e-3
}
fn default_regmean_fallback() -> bool {
    true
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

    /// Seed for the stochastic methods (DARE, DELLA). Each tensor and each
    /// model draws its own independent mask from it, so a merge is
    /// reproducible and no two task vectors share a drop pattern. Defaults
    /// to [`DEFAULT_SEED`].
    #[serde(default)]
    pub seed: Option<u64>,
}

/// The seed used when none is configured: merges are reproducible by default.
pub const DEFAULT_SEED: u64 = 0;

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

    /// Get the seed of the stochastic methods, [`DEFAULT_SEED`] unless set.
    pub fn seed(&self) -> u64 {
        self.seed.unwrap_or(DEFAULT_SEED)
    }

    /// Get gamma with default of 0.01.
    pub fn gamma(&self) -> f32 {
        self.gamma.unwrap_or(0.01)
    }

    /// These parameters with every filter-conditional setting resolved against
    /// `tensor_name`: each `weight`/`density`/`t`/`lambda` becomes a scalar
    /// (or `None` when no conditional entry matches, so the usual default
    /// applies). The merge pipelines call this once per tensor, so the
    /// methods only ever see scalars.
    pub fn resolved(&self, tensor_name: &str) -> MergeParameters {
        let resolve = |s: &Option<ParameterSetting>| {
            s.as_ref()
                .and_then(|s| s.resolve(tensor_name))
                .map(ParameterSetting::Scalar)
        };
        MergeParameters {
            weight: resolve(&self.weight),
            density: resolve(&self.density),
            t: resolve(&self.t),
            lambda: resolve(&self.lambda),
            ..self.clone()
        }
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
            seed: other.seed.or(self.seed),
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
        Ok(serde_saphyr::from_str(yaml)?)
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
        self.merge_method
            .validate_for(self.models.len(), self.base_model.is_some(), "")?;
        Ok(())
    }

    fn validate_sliced(&self) -> crate::Result<()> {
        let slices = self.slices.as_ref().expect("is_sliced() was true");

        if slices.is_empty() {
            return Err(crate::MergeError::InvalidConfig(
                "slices array must not be empty".to_string(),
            ));
        }
        if !self.models.is_empty() {
            return Err(crate::MergeError::InvalidConfig(
                "`models` and `slices` are mutually exclusive; list the models inside the slices"
                    .to_string(),
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

            // Sources are merged layer by layer, so they must cover the same
            // number of layers.
            let n_layers = slice.sources[0].n_layers();
            if let Some((src_idx, src)) = slice
                .sources
                .iter()
                .enumerate()
                .find(|(_, s)| s.n_layers() != n_layers)
            {
                return Err(crate::MergeError::InvalidConfig(format!(
                    "slice[{}].sources[{}]: layer_range covers {} layers but sources[0] covers {}",
                    slice_idx,
                    src_idx,
                    src.n_layers(),
                    n_layers
                )));
            }

            let effective_method = slice.merge_method.as_ref().unwrap_or(&self.merge_method);
            effective_method.validate_for(
                slice.sources.len(),
                slice.base_model.is_some() || self.base_model.is_some(),
                &format!("slice[{slice_idx}]: "),
            )?;
        }

        Ok(())
    }
}

impl MergeMethodConfig {
    /// Whether the method works on task vectors and so needs a base model.
    pub fn requires_base_model(&self) -> bool {
        matches!(
            self,
            Self::TaskArithmetic
                | Self::Ties
                | Self::DareTies
                | Self::DareLinear
                | Self::Della
                | Self::DellaLinear
                | Self::Breadcrumbs
                | Self::ModelStock
                | Self::Nearswap
                | Self::Ram
                | Self::RamPlus
        )
    }

    /// Check this method against the number of models it will merge and
    /// whether a base model is configured. `context` prefixes the messages.
    fn validate_for(&self, n_models: usize, has_base: bool, context: &str) -> crate::Result<()> {
        use crate::MergeError;
        if self.requires_base_model() && !has_base {
            return Err(MergeError::BaseModelRequired {
                method: format!("{context}{self:?}"),
            });
        }
        if matches!(self, Self::Slerp) && n_models != 2 {
            return Err(MergeError::InvalidConfig(format!(
                "{context}SLERP requires exactly 2 models, got {n_models}"
            )));
        }
        let side_files = match self {
            Self::Fisher { fisher_paths, .. } => Some(("fisher_paths", fisher_paths.len())),
            Self::RegMean { gram_paths, .. } => Some(("gram_paths", gram_paths.len())),
            _ => None,
        };
        if let Some((key, count)) = side_files {
            if count != n_models {
                return Err(MergeError::InvalidConfig(format!(
                    "{context}{key} lists {count} paths for {n_models} models; one per model is required"
                )));
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
            allow_mixed_dtype: false,
            sanity: crate::sanity::SanityLevel::default(),
            dry_run: false,
            align_moe_experts: false,
            vocab_mismatch: VocabMismatch::default(),
            max_shard_size: None,
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

    fn model(name: &str) -> ModelConfig {
        ModelConfig {
            model: name.to_string(),
            parameters: Default::default(),
        }
    }

    #[test]
    fn every_task_vector_method_requires_a_base_model() {
        for method in [
            MergeMethodConfig::TaskArithmetic,
            MergeMethodConfig::Nearswap,
            MergeMethodConfig::Ties,
            MergeMethodConfig::ModelStock,
            MergeMethodConfig::Breadcrumbs,
        ] {
            let config = MergeConfig {
                merge_method: method.clone(),
                models: vec![model("a"), model("b")],
                ..Default::default()
            };
            let err = config.validate().expect_err(&format!("{method:?}"));
            assert!(matches!(err, crate::MergeError::BaseModelRequired { .. }));

            let with_base = MergeConfig {
                base_model: Some("base".into()),
                ..config
            };
            with_base.validate().unwrap();
        }
    }

    #[test]
    fn side_file_paths_must_match_the_model_count() {
        let config = MergeConfig {
            merge_method: MergeMethodConfig::Fisher {
                fisher_paths: vec!["f1".into()],
                eps: 1e-8,
                fallback_to_mean: true,
            },
            models: vec![model("a"), model("b")],
            ..Default::default()
        };
        let err = config.validate().unwrap_err().to_string();
        assert!(
            err.contains("fisher_paths") && err.contains("2 models"),
            "{err}"
        );
    }

    #[test]
    fn slices_and_models_are_exclusive_and_sources_must_cover_equal_layer_counts() {
        let source = |m: &str, range: (usize, usize)| InputSlice {
            model: m.to_string(),
            layer_range: range,
            parameters: Default::default(),
        };
        let sliced = |models: Vec<ModelConfig>, sources: Vec<InputSlice>| MergeConfig {
            models,
            slices: Some(vec![OutputSlice {
                sources,
                base_model: None,
                merge_method: None,
                parameters: Default::default(),
            }]),
            ..Default::default()
        };

        let both = sliced(vec![model("a")], vec![source("a", (0, 4))]);
        assert!(
            both.validate()
                .unwrap_err()
                .to_string()
                .contains("mutually exclusive")
        );

        let uneven = sliced(vec![], vec![source("a", (0, 4)), source("b", (0, 6))]);
        let err = uneven.validate().unwrap_err().to_string();
        assert!(
            err.contains("covers 6 layers but sources[0] covers 4"),
            "{err}"
        );

        sliced(vec![], vec![source("a", (0, 4)), source("b", (8, 12))])
            .validate()
            .unwrap();
    }

    #[test]
    fn vocab_mismatch_and_shard_size_parse_from_yaml() {
        let config = MergeConfig::from_yaml(
            "merge_method: linear\nmodels:\n  - model: a\nvocab_mismatch: extend\nmax_shard_size: 1000\n",
        )
        .unwrap();
        assert_eq!(config.vocab_mismatch, VocabMismatch::Extend);
        assert_eq!(config.max_shard_size, Some(1000));
        let default =
            MergeConfig::from_yaml("merge_method: linear\nmodels:\n  - model: a\n").unwrap();
        assert_eq!(default.vocab_mismatch, VocabMismatch::Error);
        assert_eq!(default.max_shard_size, None);
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
        let params: MergeParameters = serde_saphyr::from_str(yaml).unwrap();
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
        let params: MergeParameters = serde_saphyr::from_str(yaml).unwrap();
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
