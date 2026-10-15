//! Configuration types for knowledge distillation.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Complete distillation configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillConfig {
    /// Teacher model path or HuggingFace repo ID.
    pub teacher: String,

    /// Student model path or HuggingFace repo ID.
    pub student: String,

    /// Distillation method.
    #[serde(default)]
    pub method: DistillMethod,

    /// Loss configuration.
    #[serde(default)]
    pub loss: LossConfig,

    /// Offline distillation settings.
    #[serde(default)]
    pub offline: Option<OfflineConfig>,

    /// Output path for distilled model.
    #[serde(default)]
    pub output_path: Option<PathBuf>,

    /// Training configuration.
    #[serde(default)]
    pub training: TrainingConfig,
}

/// Distillation method.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DistillMethod {
    /// Online distillation - teacher runs alongside student.
    #[default]
    Online,

    /// Offline distillation - use pre-computed teacher logits.
    Offline,

    /// Progressive distillation - gradually reduce teacher influence.
    Progressive,
}

/// Loss function configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossConfig {
    /// Primary loss type.
    #[serde(default)]
    pub loss_type: LossType,

    /// Temperature for softmax (default: 2.0).
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Alpha for blending hard and soft targets (default: 0.5).
    /// Final loss = alpha * soft_loss + (1 - alpha) * hard_loss
    #[serde(default = "default_alpha")]
    pub alpha: f32,

    /// Whether to use reverse KL (student || teacher) instead of forward KL.
    #[serde(default)]
    pub reverse_kl: bool,

    /// Whether to use reasoning-aware (rationale) distillation.
    #[serde(default)]
    pub rationale: bool,

    /// Weight for high-entropy (reasoning) tokens.
    #[serde(default = "default_rationale_weight")]
    pub rationale_weight: f32,

    /// Whether to use outcome-supervised distillation (requires correctness labels).
    #[serde(default)]
    pub outcome_supervised: bool,

    /// Start marker for explicit reasoning (e.g., `<think>`).
    pub start_marker: Option<String>,

    /// End marker for explicit reasoning (e.g., `</think>`).
    pub end_marker: Option<String>,

    /// Hidden state distillation configuration.
    #[serde(default)]
    pub hidden_state: Option<HiddenStateConfig>,

    /// Attention transfer configuration.
    #[serde(default)]
    pub attention: Option<AttentionConfig>,
}

impl Default for LossConfig {
    fn default() -> Self {
        Self {
            loss_type: LossType::default(),
            temperature: default_temperature(),
            alpha: default_alpha(),
            reverse_kl: false,
            rationale: false,
            rationale_weight: default_rationale_weight(),
            outcome_supervised: false,
            start_marker: None,
            end_marker: None,
            hidden_state: None,
            attention: None,
        }
    }
}

fn default_temperature() -> f32 {
    2.0
}

fn default_alpha() -> f32 {
    0.5
}

fn default_rationale_weight() -> f32 {
    1.0
}

/// Loss function type.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LossType {
    /// KL Divergence (forward: teacher || student).
    #[default]
    KlDivergence,

    /// Jensen-Shannon Divergence.
    JensenShannon,

    /// Cross-entropy with soft targets.
    SoftCrossEntropy,

    /// Mean Squared Error on logits.
    MseLoss,

    /// Generalized Jensen-Shannon divergence `JSD(β)` of Agarwal et al.
    /// (ICLR 2024), with `alpha` as β: `β·KL(T || M) + (1-β)·KL(S || M)` with
    /// `M = β·T + (1-β)·S`. `alpha = 0.5` is the symmetric JSD; `0` is the
    /// forward KL `KL(T || S)` and `1` the reverse KL `KL(S || T)`.
    JsdSkewed {
        /// Mixing weight β, in `[0, 1]`.
        #[serde(default = "default_jsd_skewed_alpha")]
        alpha: f32,
    },

    /// DistiLLM's skew KL (Ko et al., ICML 2024): `KL(T || α·T + (1-α)·S)`,
    /// or with `reverse` the skew reverse KL `KL(S || (1-α)·T + α·S)`. The
    /// paper's best value is `alpha = 0.1`.
    SkewKl {
        /// Mixing weight, in `[0, 1)`.
        #[serde(default = "default_skew_kl_alpha")]
        alpha: f32,
        /// Take the reverse direction.
        #[serde(default)]
        reverse: bool,
    },

    /// Universal Logit Distillation (Boizard et al. 2024). Cross-tokenizer
    /// KD via Wasserstein-1 over sorted probability vectors. Use this when
    /// teacher and student have different tokenizers and standard KL is
    /// ill-defined.
    UniversalLogit {
        /// Optional truncation to top-K entries before sorting; useful for
        /// vocabularies ≫ 32K.
        #[serde(default)]
        top_k: Option<usize>,
    },

    /// Generalized Knowledge Distillation (Agarwal et al. 2024). Loss-only
    /// variant: callers should use `GkdLoss::compute_full` directly when
    /// they have on-policy student samples available.
    Gkd {
        /// On-policy weight `λ ∈ [0, 1]`.
        #[serde(default = "default_gkd_lambda")]
        lambda: f32,
        /// Sampler temperature for the on-policy draw.
        #[serde(default = "default_gkd_sampler_temp")]
        sampler_temperature: f32,
    },
}

fn default_jsd_skewed_alpha() -> f32 {
    0.5
}
fn default_skew_kl_alpha() -> f32 {
    0.1
}
fn default_gkd_lambda() -> f32 {
    0.5
}
fn default_gkd_sampler_temp() -> f32 {
    1.0
}

/// Hidden state distillation configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiddenStateConfig {
    /// Layer mapping from teacher to student.
    /// Format: [(teacher_layer, student_layer), ...]
    #[serde(default)]
    pub layer_mapping: Vec<(usize, usize)>,

    /// Loss type for hidden states.
    #[serde(default)]
    pub loss_type: HiddenStateLossType,

    /// Weight for hidden state loss in total loss.
    #[serde(default = "default_hidden_weight")]
    pub weight: f32,

    /// Whether to use a projection layer for dimension mismatch.
    #[serde(default)]
    pub use_projection: bool,
}

fn default_hidden_weight() -> f32 {
    0.1
}

/// Hidden state loss type.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HiddenStateLossType {
    /// Mean Squared Error.
    #[default]
    Mse,

    /// Cosine similarity loss.
    Cosine,

    /// L1 loss.
    L1,
}

/// Attention transfer configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionConfig {
    /// Layer mapping for attention transfer.
    #[serde(default)]
    pub layer_mapping: Vec<(usize, usize)>,

    /// Loss type for attention maps.
    #[serde(default)]
    pub loss_type: AttentionLossType,

    /// Weight for attention loss.
    #[serde(default = "default_attention_weight")]
    pub weight: f32,

    /// Head-reduction strategy when teacher/student have different head counts
    /// or when normalising per-head maps before comparison.
    #[serde(default)]
    pub head_reduction: AttentionHeadReduction,
}

/// Attention transfer loss variants.
///
/// `Mse` and `Kl` compare teacher/student softmaxed attention matrices
/// (post-softmax probabilities, shape `[batch, heads, q_len, k_len]`).
/// `AttentionTransfer` implements the Zagoruyko & Komodakis 2017 attention-map
/// formulation (sum-of-squares across heads → L2-normalised → MSE).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionLossType {
    /// Mean Squared Error between attention probability matrices.
    #[default]
    Mse,

    /// Forward KL divergence between per-row attention distributions.
    /// Treats each query row as a distribution over keys.
    Kl,

    /// Zagoruyko & Komodakis attention-map distillation:
    /// sum across heads, L2-normalise per (batch, query), MSE.
    AttentionTransfer,
}

/// How to handle mismatched head counts between teacher and student.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionHeadReduction {
    /// Require identical head counts (fail otherwise). Default.
    #[default]
    Exact,

    /// Average attention maps across the head axis before comparison.
    /// Works for any teacher/student head count combination.
    MeanOverHeads,
}

fn default_attention_weight() -> f32 {
    0.01
}

/// Offline distillation configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OfflineConfig {
    /// Path to pre-computed logits.
    pub logits_path: PathBuf,

    /// Compression method for logits.
    #[serde(default)]
    pub compression: CompressionMethod,

    /// Top-k logits to keep (if using top-k compression).
    #[serde(default = "default_top_k")]
    pub top_k: usize,

    /// Whether to generate logits (true) or load existing (false).
    #[serde(default)]
    pub generate: bool,
}

fn default_top_k() -> usize {
    128
}

/// Compression method for offline logits.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompressionMethod {
    /// No compression - store full logits.
    #[default]
    None,

    /// Keep only top-k logits per token.
    TopK,

    /// Quantize to 8-bit.
    Int8,

    /// Quantize to 4-bit.
    Int4,
}

/// Training configuration for distillation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Batch size.
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,

    /// Learning rate.
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f32,

    /// Number of epochs.
    #[serde(default = "default_epochs")]
    pub epochs: usize,

    /// Warmup steps.
    #[serde(default)]
    pub warmup_steps: usize,

    /// Gradient accumulation steps.
    #[serde(default = "default_grad_accum")]
    pub gradient_accumulation_steps: usize,

    /// Maximum sequence length.
    #[serde(default = "default_max_seq_len")]
    pub max_seq_len: usize,

    /// Label value used to mark "ignore this token" positions in the hard-loss
    /// path. PyTorch's cross-entropy default is `-100`; HuggingFace pipelines
    /// also use `-100` for label masking. Tokens whose label equals
    /// `ignore_index` contribute zero loss and are excluded from the divisor.
    #[serde(default = "default_ignore_index")]
    pub ignore_index: i32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            batch_size: default_batch_size(),
            learning_rate: default_learning_rate(),
            epochs: default_epochs(),
            warmup_steps: 0,
            gradient_accumulation_steps: default_grad_accum(),
            max_seq_len: default_max_seq_len(),
            ignore_index: default_ignore_index(),
        }
    }
}

fn default_ignore_index() -> i32 {
    -100
}

fn default_batch_size() -> usize {
    4
}

fn default_learning_rate() -> f32 {
    2e-5
}

fn default_epochs() -> usize {
    3
}

fn default_grad_accum() -> usize {
    4
}

fn default_max_seq_len() -> usize {
    2048
}

impl DistillConfig {
    /// Load configuration from a YAML file.
    pub fn from_yaml_file(path: impl AsRef<std::path::Path>) -> crate::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Self::from_yaml(&content)
    }

    /// Parse configuration from a YAML string.
    pub fn from_yaml(yaml: &str) -> crate::Result<Self> {
        Ok(serde_saphyr::from_str(yaml)?)
    }

    /// Validate the configuration.
    pub fn validate(&self) -> crate::Result<()> {
        // `!(x > 0)` rather than `x <= 0`: NaN must not pass.
        if !(self.loss.temperature > 0.0 && self.loss.temperature.is_finite()) {
            return Err(crate::DistillError::InvalidTemperature(
                self.loss.temperature,
            ));
        }

        if !(0.0..=1.0).contains(&self.loss.alpha) {
            return Err(crate::DistillError::InvalidAlpha(self.loss.alpha));
        }

        if !(self.loss.rationale_weight >= 0.0 && self.loss.rationale_weight.is_finite()) {
            return Err(crate::DistillError::InvalidConfig(format!(
                "rationale_weight must be finite and non-negative, got {}",
                self.loss.rationale_weight
            )));
        }

        // Validate reasoning markers: if rationale is enabled and markers are used,
        // both must be provided and non-empty
        if self.loss.rationale {
            match (&self.loss.start_marker, &self.loss.end_marker) {
                (Some(start), Some(end)) => {
                    if start.is_empty() || end.is_empty() {
                        return Err(crate::DistillError::InvalidConfig(
                            "reasoning markers must be non-empty when provided".into(),
                        ));
                    }
                    if start == end {
                        return Err(crate::DistillError::InvalidConfig(
                            "start_marker and end_marker must be different".into(),
                        ));
                    }
                }
                (Some(_), None) | (None, Some(_)) => {
                    return Err(crate::DistillError::InvalidConfig(
                        "both start_marker and end_marker must be provided together".into(),
                    ));
                }
                (None, None) => {} // OK: entropy-based detection
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_basic_config() {
        let yaml = r#"
teacher: meta-llama/Llama-2-70b
student: meta-llama/Llama-2-7b
method: online
loss:
  loss_type: kl_divergence
  temperature: 2.0
  alpha: 0.5
"#;

        let config = DistillConfig::from_yaml(yaml).unwrap();
        assert_eq!(config.teacher, "meta-llama/Llama-2-70b");
        assert_eq!(config.student, "meta-llama/Llama-2-7b");
        assert!(matches!(config.method, DistillMethod::Online));
        assert_eq!(config.loss.temperature, 2.0);
    }

    /// Every loss type reaches the builder from YAML, parameters and defaults
    /// included, and builds the loss its name promises.
    #[test]
    fn test_parse_every_loss_type() {
        let cases = [
            ("kl_divergence", "kl_divergence"),
            ("jensen_shannon", "jensen_shannon"),
            ("soft_cross_entropy", "soft_cross_entropy"),
            ("mse_loss", "mse"),
            ("{ jsd_skewed: { alpha: 0.1 } }", "jsd_skewed"),
            ("{ jsd_skewed: {} }", "jsd_skewed"),
            ("{ skew_kl: { alpha: 0.1 } }", "skew_kl"),
            (
                "{ skew_kl: { alpha: 0.1, reverse: true } }",
                "skew_reverse_kl",
            ),
            ("{ universal_logit: { top_k: 64 } }", "universal_logit"),
            ("{ universal_logit: {} }", "universal_logit"),
            ("{ gkd: { lambda: 0.3 } }", "gkd"),
        ];
        for (loss_type, name) in cases {
            let yaml = format!(
                "teacher: t\nstudent: s\nloss:\n  loss_type: {loss_type}\n  temperature: 2.0\n"
            );
            let config = DistillConfig::from_yaml(&yaml)
                .unwrap_or_else(|e| panic!("`{loss_type}` does not parse: {e}"));
            let distiller = crate::Distiller::new(config).unwrap();
            assert_eq!(distiller.loss_name(), name, "`{loss_type}`");
        }
    }

    #[test]
    fn test_parse_offline_config() {
        let yaml = r#"
teacher: teacher_model
student: student_model
method: offline
offline:
  logits_path: /path/to/logits
  compression: top_k
  top_k: 64
  generate: true
"#;

        let config = DistillConfig::from_yaml(yaml).unwrap();
        assert!(matches!(config.method, DistillMethod::Offline));
        let offline = config.offline.unwrap();
        assert!(matches!(offline.compression, CompressionMethod::TopK));
        assert_eq!(offline.top_k, 64);
    }

    #[test]
    fn test_validate_temperature() {
        let config = DistillConfig {
            teacher: "t".to_string(),
            student: "s".to_string(),
            method: DistillMethod::Online,
            loss: LossConfig {
                temperature: -1.0,
                ..Default::default()
            },
            offline: None,
            output_path: None,
            training: Default::default(),
        };

        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_alpha() {
        let config = DistillConfig {
            teacher: "t".to_string(),
            student: "s".to_string(),
            method: DistillMethod::Online,
            loss: LossConfig {
                alpha: 1.5,
                ..Default::default()
            },
            offline: None,
            output_path: None,
            training: Default::default(),
        };

        assert!(config.validate().is_err());
    }
}
