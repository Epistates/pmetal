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
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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

    /// Weight for attention loss.
    #[serde(default = "default_attention_weight")]
    pub weight: f32,
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
        }
    }
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
        Ok(serde_yaml::from_str(yaml)?)
    }

    /// Validate the configuration.
    pub fn validate(&self) -> crate::Result<()> {
        if self.loss.temperature <= 0.0 {
            return Err(crate::DistillError::InvalidTemperature(
                self.loss.temperature,
            ));
        }

        if self.loss.alpha < 0.0 || self.loss.alpha > 1.0 {
            return Err(crate::DistillError::InvalidAlpha(self.loss.alpha));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

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
