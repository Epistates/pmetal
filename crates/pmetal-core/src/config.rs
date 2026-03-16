//! Configuration types for PMetal.

use crate::{Device, Dtype, Quantization};
use serde::{Deserialize, Serialize};

/// Model loading configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model identifier (HuggingFace repo ID or local path).
    pub model_id: String,

    /// Data type for model weights.
    #[serde(default)]
    pub dtype: Dtype,

    /// Quantization scheme.
    #[serde(default)]
    pub quantization: Quantization,

    /// Compute device.
    #[serde(default)]
    pub device: Device,

    /// Maximum sequence length.
    #[serde(default = "default_max_seq_len")]
    pub max_seq_len: usize,

    /// Use flash attention if available.
    #[serde(default = "default_true")]
    pub use_flash_attention: bool,

    /// Trust remote code (for custom model implementations).
    #[serde(default)]
    pub trust_remote_code: bool,

    /// Revision/branch to use.
    #[serde(default)]
    pub revision: Option<String>,

    /// HuggingFace token for private models.
    /// Skipped during serialization to prevent accidental token leakage into
    /// config snapshots, logs, or checkpoint metadata.
    #[serde(default, skip_serializing)]
    pub hf_token: Option<String>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_id: String::new(),
            dtype: Dtype::default(),
            quantization: Quantization::default(),
            device: Device::default(),
            max_seq_len: default_max_seq_len(),
            use_flash_attention: true,
            trust_remote_code: false,
            revision: None,
            hf_token: None,
        }
    }
}

/// Bias handling mode for LoRA layers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoraBias {
    /// Do not train any bias parameters (recommended default).
    #[default]
    None,
    /// Train all bias parameters.
    All,
    /// Train only bias parameters associated with LoRA layers.
    LoraOnly,
}

/// LoRA configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraConfig {
    /// LoRA rank (r).
    #[serde(default = "default_lora_r")]
    pub r: usize,

    /// LoRA alpha (scaling factor).
    #[serde(default = "default_lora_alpha")]
    pub alpha: f32,

    /// Dropout probability.
    #[serde(default)]
    pub dropout: f32,

    /// Target modules to apply LoRA to.
    #[serde(default = "default_target_modules")]
    pub target_modules: Vec<String>,

    /// Use rslora scaling.
    #[serde(default)]
    pub use_rslora: bool,

    /// Use DoRA (Weight-Decomposed Low-Rank Adaptation).
    #[serde(default)]
    pub use_dora: bool,

    /// Bias handling mode.
    #[serde(default)]
    pub bias: LoraBias,

    /// Initialize LoRA B to zero (recommended).
    #[serde(default = "default_true")]
    pub init_lora_weights: bool,

    /// LoRA+ learning rate ratio for B matrices (Hayou et al., ICML 2024).
    ///
    /// When set, LoRA B matrices are trained with `base_lr * loraplus_lr_ratio`
    /// while LoRA A matrices use `base_lr`.  This breaks the symmetry between A and B,
    /// letting B learn faster since it starts at zero and directly controls the output.
    ///
    /// Recommended value: 16.0 (from the paper).  `None` disables LoRA+ (default).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loraplus_lr_ratio: Option<f32>,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            r: default_lora_r(),
            alpha: default_lora_alpha(),
            dropout: 0.0,
            target_modules: default_target_modules(),
            use_rslora: false,
            use_dora: false,
            bias: LoraBias::default(),
            init_lora_weights: true,
            loraplus_lr_ratio: None,
        }
    }
}

impl LoraConfig {
    /// Compute the LoRA scaling factor.
    #[must_use]
    pub fn scaling(&self) -> f32 {
        if self.r == 0 {
            return 0.0;
        }

        if self.use_rslora {
            self.alpha / (self.r as f32).sqrt()
        } else {
            self.alpha / self.r as f32
        }
    }
}

/// Training configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Learning rate.
    #[serde(default = "default_lr")]
    pub learning_rate: f64,

    /// Separate learning rate for embedding layers.
    /// If set, embedding parameters use this learning rate instead of the base learning_rate.
    /// Typically set lower than the base LR (e.g., 5e-5 for embeddings vs 2e-4 for LoRA).
    #[serde(default)]
    pub embedding_learning_rate: Option<f64>,

    /// Batch size per device.
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,

    /// Gradient accumulation steps.
    #[serde(default = "default_gradient_accumulation_steps")]
    pub gradient_accumulation_steps: usize,

    /// Number of training epochs.
    #[serde(default = "default_epochs")]
    pub num_epochs: usize,

    /// Maximum training steps (overrides epochs if set).
    #[serde(default)]
    pub max_steps: Option<usize>,

    /// Warmup steps.
    #[serde(default = "default_warmup")]
    pub warmup_steps: usize,

    /// Warmup ratio (alternative to warmup_steps).
    #[serde(default)]
    pub warmup_ratio: Option<f64>,

    /// Weight decay.
    #[serde(default = "default_weight_decay")]
    pub weight_decay: f64,

    /// Maximum gradient norm for clipping.
    #[serde(default = "default_grad_clip")]
    pub max_grad_norm: f64,

    /// Learning rate scheduler type.
    #[serde(default)]
    pub lr_scheduler: LrSchedulerType,

    /// Minimum learning rate floor (absolute). Applied across all decay schedulers.
    /// Without this, cosine/polynomial decay goes all the way to zero.
    #[serde(default)]
    pub min_lr: Option<f64>,

    /// WSD stable phase fraction (0.0-1.0, default 0.7).
    #[serde(default)]
    pub wsd_stable_ratio: Option<f64>,

    /// Cosine restart count for CosineWithRestarts scheduler (default 1).
    #[serde(default)]
    pub cosine_num_restarts: Option<usize>,

    /// Polynomial decay exponent (default 1.0 = linear, 2.0 = quadratic).
    #[serde(default)]
    pub polynomial_power: Option<f64>,

    /// Gradient checkpointing strategy.
    #[serde(default)]
    pub gradient_checkpointing: CheckpointStrategy,

    /// Optimizer type.
    #[serde(default)]
    pub optimizer: OptimizerType,

    /// Random seed.
    #[serde(default = "default_seed")]
    pub seed: u64,

    /// Logging steps.
    #[serde(default = "default_logging_steps")]
    pub logging_steps: usize,

    /// Evaluation steps.
    #[serde(default)]
    pub eval_steps: Option<usize>,

    /// Save steps.
    #[serde(default)]
    pub save_steps: Option<usize>,

    /// Output directory.
    #[serde(default = "default_output_dir")]
    pub output_dir: String,

    /// Use packing for efficient training.
    #[serde(default = "default_true")]
    pub use_packing: bool,

    /// Maximum sequence length.
    #[serde(default = "default_max_seq_len")]
    pub max_seq_len: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: default_lr(),
            embedding_learning_rate: None,
            batch_size: default_batch_size(),
            gradient_accumulation_steps: default_gradient_accumulation_steps(),
            num_epochs: default_epochs(),
            max_steps: None,
            warmup_steps: default_warmup(),
            warmup_ratio: None,
            weight_decay: default_weight_decay(),
            max_grad_norm: default_grad_clip(),
            lr_scheduler: LrSchedulerType::default(),
            min_lr: None,
            wsd_stable_ratio: None,
            cosine_num_restarts: None,
            polynomial_power: None,
            gradient_checkpointing: CheckpointStrategy::default(),
            optimizer: OptimizerType::default(),
            seed: default_seed(),
            logging_steps: default_logging_steps(),
            eval_steps: None,
            save_steps: None,
            output_dir: default_output_dir(),
            use_packing: true,
            max_seq_len: default_max_seq_len(),
        }
    }
}

/// Learning rate scheduler type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum LrSchedulerType {
    /// Constant learning rate.
    Constant,
    /// Linear decay.
    Linear,
    /// Cosine annealing.
    #[default]
    Cosine,
    /// Cosine with restarts.
    CosineWithRestarts,
    /// Polynomial decay.
    Polynomial,
    /// Warmup-Stable-Decay: linear warmup → constant plateau → linear decay.
    /// Modern default for LLM training. Stable phase ratio defaults to 0.7.
    Wsd,
}

/// Gradient checkpointing strategy.
///
/// **Not yet implemented for the MLX backend** — selecting a strategy other than
/// `None` has no effect on peak memory usage. The option is retained so configs
/// remain forward-compatible once backend support lands.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointStrategy {
    /// No checkpointing (default — gradient checkpointing is not yet implemented).
    #[default]
    None,
    /// Checkpoint every N layers.
    EveryN(usize),
    /// Smart checkpointing based on memory budget.
    Smart,
    /// Selective attention-only checkpointing.
    SelectiveAttention,
}

/// Optimizer type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum OptimizerType {
    /// AdamW optimizer.
    #[default]
    AdamW,
    /// SGD with momentum.
    Sgd,
    /// Adafactor (memory-efficient).
    Adafactor,
    /// Lion optimizer.
    Lion,
}

/// Dataset configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    /// Dataset identifier (HuggingFace or local path).
    pub dataset_id: String,

    /// Dataset split to use.
    #[serde(default = "default_split")]
    pub split: String,

    /// Column containing input text.
    #[serde(default = "default_text_column")]
    pub text_column: String,

    /// Maximum samples to use (None for all).
    #[serde(default)]
    pub max_samples: Option<usize>,

    /// Shuffle the dataset.
    #[serde(default = "default_true")]
    pub shuffle: bool,

    /// Random seed for shuffling.
    #[serde(default = "default_seed")]
    pub seed: u64,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            dataset_id: String::new(),
            split: default_split(),
            text_column: default_text_column(),
            max_samples: None,
            shuffle: true,
            seed: default_seed(),
        }
    }
}

// Default value functions
fn default_max_seq_len() -> usize {
    8192
}
fn default_true() -> bool {
    true
}
fn default_lora_r() -> usize {
    16
}
fn default_lora_alpha() -> f32 {
    32.0
}
fn default_target_modules() -> Vec<String> {
    vec![
        "q_proj".into(),
        "k_proj".into(),
        "v_proj".into(),
        "o_proj".into(),
    ]
}
fn default_lr() -> f64 {
    2e-4
}
fn default_batch_size() -> usize {
    1
}
fn default_gradient_accumulation_steps() -> usize {
    4
}
fn default_epochs() -> usize {
    3
}
fn default_warmup() -> usize {
    100
}
fn default_weight_decay() -> f64 {
    0.01
}
fn default_grad_clip() -> f64 {
    1.0
}
fn default_seed() -> u64 {
    42
}
fn default_logging_steps() -> usize {
    10
}

fn default_output_dir() -> String {
    "./output".into()
}
fn default_split() -> String {
    "train".into()
}
fn default_text_column() -> String {
    "text".into()
}

#[cfg(test)]
mod tests {
    use super::LoraConfig;

    #[test]
    fn lora_scaling_is_zero_for_zero_rank() {
        let config = LoraConfig {
            r: 0,
            alpha: 32.0,
            ..Default::default()
        };

        assert_eq!(config.scaling(), 0.0);
    }
}
