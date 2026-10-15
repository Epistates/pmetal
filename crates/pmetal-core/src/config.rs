//! Configuration types for PMetal.
//!
//! These are the sections of a training config file (`model`, `lora`,
//! `training`, `dataset`) and the library types the trainers take. Every field
//! here is one a trainer reads: a setting nothing honours is a setting that
//! silently does nothing, so the sections reject unknown keys instead, and
//! [`ConfigValidator`](crate::ConfigValidator) rejects the values (and the
//! combinations) that cannot work. See [`crate::FullTrainingConfig`] for the
//! file loader, which also keeps files written by earlier releases readable.

use crate::defaults;
use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Model loading configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    /// Model identifier (HuggingFace repo ID or local path).
    #[serde(default, alias = "name")]
    pub model_id: String,

    /// Use flash attention if available.
    #[serde(default = "default_true")]
    pub use_flash_attention: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_id: String::new(),
            use_flash_attention: true,
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

/// How a LoRA adapter's `A` and `B` matrices are initialised.
///
/// In a config file this is the `init_lora_weights` key: `true` (or
/// `standard`) for the default, or a method's name. Only
/// [`Standard`](Self::Standard) is implemented; the others parse so that a
/// config naming one is refused with a clear error
/// ([`ConfigValidator`](crate::ConfigValidator)) instead of being trained as
/// something else.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum LoraInit {
    /// `A` Kaiming-uniform, `B` zero: the adapter starts as the identity
    /// (Hu et al., 2021).
    #[default]
    Standard,
    /// `A` and `B` from the principal singular components of the base weight,
    /// whose residual becomes the frozen base (Meng et al., 2024).
    Pissa,
    /// `A` and `B` from an SVD of the quantization residual, with the base
    /// quantized alternately (Li et al., 2023). For QLoRA.
    Loftq,
}

impl LoraInit {
    /// Every method, implemented or not.
    pub const ALL: [Self; 3] = [Self::Standard, Self::Pissa, Self::Loftq];

    /// The name a config file uses.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::Pissa => "pissa",
            Self::Loftq => "loftq",
        }
    }

    /// Whether the trainers can initialise an adapter this way.
    #[must_use]
    pub fn is_supported(self) -> bool {
        matches!(self, Self::Standard)
    }
}

impl std::fmt::Display for LoraInit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for LoraInit {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let wanted = s.trim().to_ascii_lowercase();
        if wanted == "true" || wanted == "default" {
            return Ok(Self::Standard);
        }
        Self::ALL
            .into_iter()
            .find(|init| init.as_str() == wanted)
            .ok_or_else(|| {
                let valid: Vec<_> = Self::ALL.iter().map(|i| i.as_str()).collect();
                format!(
                    "unknown LoRA initialisation '{s}'; valid: {}",
                    valid.join(", ")
                )
            })
    }
}

impl Serialize for LoraInit {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            // The spelling files written by earlier releases carry.
            Self::Standard => serializer.serialize_bool(true),
            other => serializer.serialize_str(other.as_str()),
        }
    }
}

impl<'de> Deserialize<'de> for LoraInit {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct InitVisitor;

        impl Visitor<'_> for InitVisitor {
            type Value = LoraInit;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("`true` or the name of a LoRA initialisation (standard, pissa, loftq)")
            }

            fn visit_bool<E: de::Error>(self, v: bool) -> Result<LoraInit, E> {
                if v {
                    Ok(LoraInit::Standard)
                } else {
                    Err(E::custom(
                        "init_lora_weights: false (a randomly initialised `B`, which starts the \
                         adapter away from the base model) is not supported; use `true` or a \
                         named initialisation",
                    ))
                }
            }

            fn visit_str<E: de::Error>(self, v: &str) -> Result<LoraInit, E> {
                v.parse().map_err(E::custom)
            }
        }

        deserializer.deserialize_any(InitVisitor)
    }
}

/// LoRA configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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

    /// Projections to adapt, by name (`q_proj`, `gate_proj`, ...). Empty
    /// adapts every linear layer except the output layer (the LM head); name
    /// `lm_head` to adapt the head.
    #[serde(default = "default_target_modules")]
    pub target_modules: Vec<String>,

    /// Use rslora scaling.
    #[serde(default)]
    pub use_rslora: bool,

    /// Use DoRA (Weight-Decomposed Low-Rank Adaptation).
    #[serde(default)]
    pub use_dora: bool,

    /// Bias handling mode. Only [`LoraBias::None`] is implemented.
    #[serde(default)]
    pub bias: LoraBias,

    /// How the adapter matrices start out. Written `init_lora_weights` in a
    /// config file. Only [`LoraInit::Standard`] is implemented.
    #[serde(default)]
    pub init_lora_weights: LoraInit,

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
            init_lora_weights: LoraInit::default(),
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
#[serde(deny_unknown_fields)]
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
    #[serde(default = "default_epochs", alias = "epochs")]
    pub num_epochs: usize,

    /// Maximum training steps (overrides epochs if set).
    #[serde(default)]
    pub max_steps: Option<usize>,

    /// Warmup steps (default 0, as on the CLI and in the reference
    /// trainers). Ignored when `warmup_ratio` is set.
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
    #[serde(default, alias = "lr_schedule")]
    pub lr_scheduler: LrSchedulerType,

    /// Minimum learning rate floor (absolute). Applied across all decay schedulers.
    /// Without this, cosine/polynomial decay goes all the way to zero.
    #[serde(default)]
    pub min_lr: Option<f64>,

    /// WSD stable phase fraction (0.0-1.0, default 0.7).
    #[serde(default)]
    pub wsd_stable_ratio: Option<f64>,

    /// Shape of the WSD decay phase (default linear).
    #[serde(default)]
    pub wsd_decay_style: Option<WsdDecayStyle>,

    /// Cosine cycles for the CosineWithRestarts scheduler (default 1, which
    /// is a single cosine; `n` cycles restart `n - 1` times). transformers
    /// calls this `num_cycles`.
    #[serde(default, alias = "cosine_num_cycles")]
    pub cosine_num_restarts: Option<usize>,

    /// Polynomial decay exponent (default 1.0 = linear, 2.0 = quadratic).
    #[serde(default)]
    pub polynomial_power: Option<f64>,

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

    /// Maximum sequence length; 0 takes it from the model's config.
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
            wsd_decay_style: None,
            cosine_num_restarts: None,
            polynomial_power: None,
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

impl LrSchedulerType {
    /// Every schedule, in the order the CLI lists them.
    pub const ALL: [Self; 6] = [
        Self::Cosine,
        Self::Linear,
        Self::Constant,
        Self::CosineWithRestarts,
        Self::Polynomial,
        Self::Wsd,
    ];

    /// The name the CLI, job specs and YAML configs use.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Constant => "constant",
            Self::Linear => "linear",
            Self::Cosine => "cosine",
            Self::CosineWithRestarts => "cosine_with_restarts",
            Self::Polynomial => "polynomial",
            Self::Wsd => "wsd",
        }
    }
}

impl std::fmt::Display for LrSchedulerType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for LrSchedulerType {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let wanted = normalized_name(s);
        Self::ALL
            .into_iter()
            .find(|kind| normalized_name(kind.as_str()) == wanted)
            .ok_or_else(|| {
                let valid: Vec<_> = Self::ALL.iter().map(|k| k.as_str()).collect();
                format!("unknown LR schedule '{s}'; valid: {}", valid.join(", "))
            })
    }
}

/// Shape of the WSD schedule's final decay phase.
///
/// Hägele et al. (2024, "Scaling Laws and Compute-Optimal Training Beyond
/// Fixed Training Durations") find `1 - sqrt` cooldowns reach a lower loss than
/// linear ones at equal length; `Linear` stays the default so existing runs
/// keep their curve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum WsdDecayStyle {
    /// `1 - p`.
    #[default]
    Linear,
    /// `0.5 * (1 + cos(pi * p))`.
    Cosine,
    /// `1 - sqrt(p)`.
    #[serde(alias = "1-sqrt", alias = "1_sqrt", alias = "sqrt")]
    OneMinusSqrt,
}

impl WsdDecayStyle {
    /// The decay factor, 1 at the start of the phase and 0 at its end, for
    /// progress `p` through it (0 to 1).
    #[must_use]
    pub fn factor(self, p: f64) -> f64 {
        match self {
            Self::Linear => 1.0 - p,
            Self::Cosine => 0.5 * (1.0 + (std::f64::consts::PI * p).cos()),
            Self::OneMinusSqrt => 1.0 - p.sqrt(),
        }
    }
}

/// Optimizer type.
///
/// Every training loop runs the optimizer this names; see
/// `pmetal_trainer::optimizer` for the update rules and their defaults. The
/// learning rate and weight decay do not carry over between kinds: the Lion
/// paper recommends a learning rate 3-10x smaller and a weight decay 3-10x
/// larger than AdamW's.
///
/// [`SUPPORTED`](Self::SUPPORTED) lists the kinds a trainer runs today. The
/// rest parse, so a config naming one gets a clear refusal from
/// [`ConfigValidator`](crate::ConfigValidator) rather than an "unknown
/// optimizer".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum OptimizerType {
    /// AdamW (decoupled weight decay).
    #[serde(rename = "adamw", alias = "adam_w")]
    #[default]
    AdamW,
    /// SGD with momentum 0.9.
    Sgd,
    /// Adafactor with factored second moments and an external learning rate.
    Adafactor,
    /// Lion (sign of an interpolated momentum).
    Lion,
    /// Muon on the hidden weight matrices (momentum, then Newton-Schulz
    /// orthogonalization; Jordan et al., 2024) and AdamW on everything else
    /// (embeddings, output head, norms, biases), the recipe it is meant to be
    /// trained with.
    Muon,
    /// Schedule-free AdamW (Defazio et al., 2024): iterate averaging in place
    /// of a decaying learning-rate schedule.
    #[serde(rename = "adamw_schedule_free", alias = "schedule_free_adamw")]
    AdamWScheduleFree,
    /// AdamW with its two moment buffers blockwise-quantized to 8 bits
    /// (Dettmers et al., 2022).
    #[serde(rename = "adamw_8bit", alias = "adamw8bit")]
    AdamW8bit,
}

impl OptimizerType {
    /// Every kind, in the order the CLI lists them.
    pub const ALL: [Self; 7] = [
        Self::AdamW,
        Self::Sgd,
        Self::Lion,
        Self::Adafactor,
        Self::Muon,
        Self::AdamWScheduleFree,
        Self::AdamW8bit,
    ];

    /// The kinds a trainer runs today, in the order the CLI lists them.
    pub const SUPPORTED: [Self; 4] = [Self::AdamW, Self::Sgd, Self::Lion, Self::Adafactor];

    /// The name the CLI, job specs and YAML configs use.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AdamW => "adamw",
            Self::Sgd => "sgd",
            Self::Lion => "lion",
            Self::Adafactor => "adafactor",
            Self::Muon => "muon",
            Self::AdamWScheduleFree => "adamw_schedule_free",
            Self::AdamW8bit => "adamw_8bit",
        }
    }

    /// Whether a trainer can run this optimizer. Flip a kind's arm here when
    /// the trainer learns to build it; every surface that lists optimizers
    /// follows [`SUPPORTED`](Self::SUPPORTED).
    #[must_use]
    pub fn is_supported(self) -> bool {
        match self {
            Self::AdamW | Self::Sgd | Self::Lion | Self::Adafactor => true,
            Self::Muon | Self::AdamWScheduleFree | Self::AdamW8bit => false,
        }
    }
}

impl std::fmt::Display for OptimizerType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for OptimizerType {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let wanted = normalized_name(s);
        let wanted = match wanted.as_str() {
            "schedulefreeadamw" => "adamwschedulefree".to_string(),
            _ => wanted,
        };
        Self::ALL
            .into_iter()
            .find(|kind| normalized_name(kind.as_str()) == wanted)
            .ok_or_else(|| {
                let valid: Vec<_> = Self::ALL.iter().map(|k| k.as_str()).collect();
                format!("unknown optimizer '{s}'; valid: {}", valid.join(", "))
            })
    }
}

/// A name with case, `_` and `-` removed, for matching user spellings.
fn normalized_name(name: &str) -> String {
    name.trim().to_ascii_lowercase().replace(['_', '-'], "")
}

/// Compression strategy for distributed gradient synchronization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum DistributedCompression {
    /// No compression (full f32 gradients).
    #[default]
    None,
    /// Keep top-k% gradients by magnitude (default 1%).
    TopK,
    /// Quantize gradients to fp16.
    Fp16,
    /// Random sparsification.
    Random,
}

/// Configuration for distributed training across multiple Apple Silicon devices.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DistributedTrainingConfig {
    /// Manual peer addresses (ip:port). If empty, uses auto-discovery.
    #[serde(default)]
    pub peers: Vec<String>,

    /// Enable mDNS auto-discovery of peers on the local network.
    #[serde(default)]
    pub auto_discover: bool,

    /// Port for gradient synchronization (default: 52416).
    #[serde(default = "default_gradient_port")]
    pub gradient_port: u16,

    /// Gradient compression strategy.
    #[serde(default)]
    pub compression: DistributedCompression,

    /// Top-k ratio when using TopK compression (0.0-1.0, default 0.01 = 1%).
    #[serde(default = "default_topk_ratio")]
    pub topk_ratio: f32,

    /// Enable error feedback for lossy compression (accumulates residuals).
    #[serde(default = "default_true")]
    pub error_feedback: bool,
}

impl Default for DistributedTrainingConfig {
    fn default() -> Self {
        Self {
            peers: Vec::new(),
            auto_discover: false,
            gradient_port: default_gradient_port(),
            compression: DistributedCompression::None,
            topk_ratio: default_topk_ratio(),
            error_feedback: true,
        }
    }
}

fn default_gradient_port() -> u16 {
    52416
}

fn default_topk_ratio() -> f32 {
    0.01
}

/// Dataset configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetConfig {
    /// Dataset identifier (HuggingFace or local path).
    #[serde(default, alias = "path")]
    pub dataset_id: String,

    /// Shuffle the dataset.
    #[serde(default = "default_true")]
    pub shuffle: bool,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            dataset_id: String::new(),
            shuffle: true,
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
    defaults::LORA_R
}
fn default_lora_alpha() -> f32 {
    defaults::LORA_ALPHA
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
    defaults::LEARNING_RATE
}
fn default_batch_size() -> usize {
    defaults::CLI_BATCH_SIZE
}
fn default_gradient_accumulation_steps() -> usize {
    defaults::GRADIENT_ACCUMULATION_STEPS
}
fn default_epochs() -> usize {
    defaults::EPOCHS
}
fn default_warmup() -> usize {
    0
}
fn default_weight_decay() -> f64 {
    defaults::WEIGHT_DECAY
}
fn default_grad_clip() -> f64 {
    defaults::MAX_GRAD_NORM
}
fn default_seed() -> u64 {
    defaults::SEED
}
fn default_logging_steps() -> usize {
    10
}

fn default_output_dir() -> String {
    "./output".into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lora_scaling_is_zero_for_zero_rank() {
        let config = LoraConfig {
            r: 0,
            alpha: 32.0,
            ..Default::default()
        };

        assert_eq!(config.scaling(), 0.0);
    }

    #[test]
    fn optimizer_names_round_trip_through_the_cli_and_yaml() {
        for kind in OptimizerType::ALL {
            assert_eq!(kind.as_str().parse::<OptimizerType>(), Ok(kind));
            let yaml = serde_saphyr::to_string(&kind).unwrap();
            assert_eq!(yaml.trim(), kind.as_str());
            let back: OptimizerType = serde_saphyr::from_str(&yaml).unwrap();
            assert_eq!(back, kind);
        }
        assert_eq!("AdamW".parse::<OptimizerType>(), Ok(OptimizerType::AdamW));
        assert_eq!("adam_w".parse::<OptimizerType>(), Ok(OptimizerType::AdamW));
        assert_eq!(
            "schedule-free-adamw".parse::<OptimizerType>(),
            Ok(OptimizerType::AdamWScheduleFree)
        );
        assert_eq!(
            "adamw8bit".parse::<OptimizerType>(),
            Ok(OptimizerType::AdamW8bit)
        );
        assert!("adam".parse::<OptimizerType>().is_err());
        let config: TrainingConfig = serde_saphyr::from_str("optimizer: adam_w").unwrap();
        assert_eq!(config.optimizer, OptimizerType::AdamW);
        let config: TrainingConfig = serde_saphyr::from_str("optimizer: lion").unwrap();
        assert_eq!(config.optimizer, OptimizerType::Lion);
    }

    #[test]
    fn supported_optimizers_are_the_ones_marked_supported() {
        let marked: Vec<_> = OptimizerType::ALL
            .into_iter()
            .filter(|k| k.is_supported())
            .collect();
        assert_eq!(marked, OptimizerType::SUPPORTED);
    }

    #[test]
    fn schedule_names_round_trip_and_unknown_names_are_errors() {
        for kind in LrSchedulerType::ALL {
            assert_eq!(kind.as_str().parse::<LrSchedulerType>(), Ok(kind));
            let yaml = serde_saphyr::to_string(&kind).unwrap();
            assert_eq!(yaml.trim(), kind.as_str());
        }
        assert_eq!(
            "Cosine-With-Restarts".parse::<LrSchedulerType>(),
            Ok(LrSchedulerType::CosineWithRestarts)
        );
        let err = "cosin".parse::<LrSchedulerType>().unwrap_err();
        assert!(err.contains("cosin") && err.contains("wsd"), "{err}");
    }

    #[test]
    fn lora_init_reads_true_or_a_method_name() {
        let init = |text: &str| -> Result<LoraConfig, _> {
            serde_saphyr::from_str(&format!("init_lora_weights: {text}"))
        };
        assert_eq!(init("true").unwrap().init_lora_weights, LoraInit::Standard);
        assert_eq!(init("pissa").unwrap().init_lora_weights, LoraInit::Pissa);
        assert_eq!(init("loftq").unwrap().init_lora_weights, LoraInit::Loftq);
        assert!(init("false").is_err(), "random B is not a PMetal mode");
        assert!(init("gaussian").is_err());
        // The default serializes the way files from earlier releases spell it.
        let yaml = serde_saphyr::to_string(&LoraConfig::default()).unwrap();
        assert!(yaml.contains("init_lora_weights: true"), "{yaml}");
        let back: LoraConfig = serde_saphyr::from_str(&yaml).unwrap();
        assert_eq!(back.init_lora_weights, LoraInit::Standard);
    }

    #[test]
    fn sections_reject_keys_they_do_not_have() {
        // A misspelt key used to be dropped, leaving the default in force.
        let err = serde_saphyr::from_str::<TrainingConfig>("learning_rat: 1e-4")
            .unwrap_err()
            .to_string();
        assert!(err.contains("learning_rat"), "{err}");
        assert!(serde_saphyr::from_str::<LoraConfig>("rank: 8").is_err());
        assert!(serde_saphyr::from_str::<DatasetConfig>("max_samples: 10").is_err());
    }

    #[test]
    fn documented_spellings_are_accepted() {
        let training: TrainingConfig =
            serde_saphyr::from_str("epochs: 5\nlr_schedule: wsd\ncosine_num_cycles: 3").unwrap();
        assert_eq!(training.num_epochs, 5);
        assert_eq!(training.lr_scheduler, LrSchedulerType::Wsd);
        assert_eq!(training.cosine_num_restarts, Some(3));
        let model: ModelConfig = serde_saphyr::from_str("name: org/model").unwrap();
        assert_eq!(model.model_id, "org/model");
        let dataset: DatasetConfig = serde_saphyr::from_str("path: ./train.jsonl").unwrap();
        assert_eq!(dataset.dataset_id, "./train.jsonl");
    }
}
