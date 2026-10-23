//! Clap argument struct for `pmetal distill`.

use clap::Args;
use pmetal_core::jobs::DistillSpec;

/// Knowledge distillation from a teacher into a LoRA-adapted student.
#[derive(Args, Debug)]
pub struct DistillArgs {
    /// Teacher model ID or path
    #[arg(short, long = "teacher")]
    pub teacher: String,

    /// Student model ID or path
    #[arg(short, long = "student")]
    pub student: String,

    /// Dataset path (JSONL file)
    #[arg(short, long = "dataset")]
    pub dataset: String,

    /// Optional evaluation dataset path (JSONL file)
    #[arg(long = "eval-dataset")]
    pub eval_dataset: Option<String>,

    /// Output directory for distilled student
    #[arg(short, long = "output", default_value = pmetal_core::defaults::DISTILL_OUTPUT_DIR)]
    pub output: String,

    /// Distillation method
    #[arg(long = "method", value_enum, default_value_t = DistillMethodArg::Online)]
    pub method: DistillMethodArg,

    /// Shortcut for `--method offline`.
    #[arg(long = "offline")]
    pub offline: bool,

    /// Directory used to store or load cached teacher logits for offline distillation.
    #[arg(long = "offline-cache")]
    pub offline_cache: Option<String>,

    /// Force generation of missing teacher logits for offline distillation.
    #[arg(long = "offline-generate")]
    pub offline_generate: bool,

    /// Compression used for newly created offline logit caches.
    #[arg(long = "offline-compression", value_enum, default_value_t = OfflineCompressionArg::TopK)]
    pub offline_compression: OfflineCompressionArg,

    /// Top-k width for newly created offline logit caches.
    #[arg(long = "offline-top-k", default_value_t = 128)]
    pub offline_top_k: usize,

    /// Loss: forward or reverse KL, Jensen-Shannon, soft cross-entropy, MSE on
    /// the logits, the skewed JSD (with --on-policy it is GKD) or skew KL of
    /// DistiLLM, or universal_logit for a teacher with another tokenizer
    #[arg(long = "loss-type", value_enum, default_value_t = LossTypeArg::KlDivergence)]
    pub loss_type: LossTypeArg,

    /// Mixing weight of jsd_skewed (default 0.5, the symmetric JSD) and of
    /// skew_kl / skew_reverse_kl (default 0.1)
    #[arg(long = "skew-alpha")]
    pub skew_alpha: Option<f32>,

    /// Entries universal_logit keeps before sorting (default: the whole vocabulary)
    #[arg(long = "universal-logit-top-k")]
    pub universal_logit_top_k: Option<usize>,

    /// Softmax temperature
    #[arg(long = "temperature", default_value_t = 2.0)]
    pub temperature: f32,

    /// Alpha for blending hard/soft targets (0.0 to 1.0)
    #[arg(long = "alpha", default_value = "0.5")]
    pub alpha: f32,

    /// Use reasoning-aware (rationale) distillation
    #[arg(long = "rationale")]
    pub rationale: bool,

    /// Weight for reasoning tokens (if rationale is enabled)
    #[arg(long = "rationale-weight", default_value = "1.0")]
    pub rationale_weight: f32,

    /// Share of batches whose responses the student samples itself and the
    /// teacher scores (on-policy distillation, 0.0 to 1.0); online and
    /// progressive runs only
    #[arg(long = "on-policy", default_value = "0.0")]
    pub on_policy: f32,

    /// Longest response the student samples for an on-policy batch
    #[arg(long = "on-policy-max-tokens", default_value_t = 256)]
    pub on_policy_max_tokens: usize,

    /// LoRA rank for student
    #[arg(long = "lora-r", default_value = "16")]
    pub lora_r: usize,

    /// LoRA alpha scaling factor
    #[arg(long = "lora-alpha", default_value = "32")]
    pub lora_alpha: f32,

    /// Learning rate
    #[arg(long = "learning-rate", default_value = "2e-5")]
    pub learning_rate: f32,

    /// Optimizer: adamw, sgd, lion, adafactor, muon, adamw_schedule_free or
    /// adamw_8bit. Lion wants a 3-10x smaller learning rate and 3-10x larger
    /// weight decay than AdamW; schedule-free AdamW wants a constant rate.
    #[arg(long = "optimizer", default_value = "adamw")]
    pub optimizer: pmetal_core::OptimizerType,

    /// Batch size
    #[arg(long = "batch-size", default_value = "1")]
    pub batch_size: usize,

    /// Number of epochs
    #[arg(long = "epochs", default_value = "1")]
    pub epochs: usize,

    /// Maximum sequence length
    #[arg(long = "max-seq-len", default_value = "1024")]
    pub max_seq_len: usize,

    /// Random seed for dataset shuffling and initialization.
    #[arg(long = "seed", default_value = "42")]
    pub seed: u64,

    /// Save checkpoint every N steps (0 disables interval checkpoints).
    #[arg(long = "checkpoint-every", default_value = "100")]
    pub checkpoint_every: usize,

    /// Resume from the latest checkpoint in the output directory.
    #[arg(long = "resume")]
    pub resume: bool,

    /// Custom text column name in the dataset JSONL.
    #[arg(long = "text-column")]
    pub text_column: Option<String>,

    /// Comma-separated list of columns to concatenate as the text field.
    #[arg(long = "text-columns", value_delimiter = ',')]
    pub text_columns: Option<Vec<String>>,

    /// Separator used when joining multiple text columns (default: a blank line).
    #[arg(long = "column-separator")]
    pub column_separator: Option<String>,

    /// Column name for the prompt portion (enables SFT label masking).
    #[arg(long = "prompt-column")]
    pub prompt_column: Option<String>,

    /// Column name for the response portion (enables SFT label masking).
    #[arg(long = "response-column")]
    pub response_column: Option<String>,

    /// Write per-step metrics as JSONL to this file (a relative path is relative to --output).
    #[arg(long = "log-metrics")]
    pub log_metrics: Option<String>,
}

/// The distillation method.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum DistillMethodArg {
    /// The teacher runs beside the student
    Online,
    /// Teacher logits cached on disk first
    Offline,
    /// The teacher's influence is reduced over the run
    Progressive,
}

impl DistillMethodArg {
    fn as_str(self) -> &'static str {
        match self {
            Self::Online => "online",
            Self::Offline => "offline",
            Self::Progressive => "progressive",
        }
    }
}

/// How an offline logit cache is stored.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
#[value(rename_all = "snake_case")]
pub enum OfflineCompressionArg {
    /// The top-k logits of each position
    TopK,
    /// Every logit
    None,
    /// 8-bit quantized
    Int8,
    /// 4-bit quantized
    Int4,
    /// Draws from the teacher's distribution
    Sampled,
}

impl OfflineCompressionArg {
    fn as_str(self) -> &'static str {
        match self {
            Self::TopK => "top_k",
            Self::None => "none",
            Self::Int8 => "int8",
            Self::Int4 => "int4",
            Self::Sampled => "sampled",
        }
    }
}

/// The distillation loss.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
#[value(rename_all = "snake_case")]
pub enum LossTypeArg {
    /// Forward KL, teacher to student
    KlDivergence,
    /// Reverse KL, student to teacher
    ReverseKl,
    /// Jensen-Shannon divergence
    JensenShannon,
    /// Cross-entropy against the teacher's soft targets
    SoftCrossEntropy,
    /// Mean squared error on the logits
    MseLoss,
    /// Skewed generalized JSD (GKD's divergence), mixing weight --skew-alpha
    JsdSkewed,
    /// DistiLLM's skew KL, mixing weight --skew-alpha
    SkewKl,
    /// DistiLLM's skew reverse KL, mixing weight --skew-alpha
    SkewReverseKl,
    /// Universal logit distillation, for a teacher with another tokenizer
    UniversalLogit,
}

impl LossTypeArg {
    fn as_str(self) -> &'static str {
        match self {
            Self::KlDivergence => "kl_divergence",
            Self::ReverseKl => "reverse_kl",
            Self::JensenShannon => "jensen_shannon",
            Self::SoftCrossEntropy => "soft_cross_entropy",
            Self::MseLoss => "mse_loss",
            Self::JsdSkewed => "jsd_skewed",
            Self::SkewKl => "skew_kl",
            Self::SkewReverseKl => "skew_reverse_kl",
            Self::UniversalLogit => "universal_logit",
        }
    }
}

impl From<DistillArgs> for DistillSpec {
    fn from(a: DistillArgs) -> Self {
        DistillSpec {
            teacher: a.teacher,
            student: a.student,
            dataset: a.dataset,
            eval_dataset: a.eval_dataset,
            output_dir: a.output,
            method: a.method.as_str().to_string(),
            offline: a.offline,
            offline_cache: a.offline_cache,
            offline_generate: a.offline_generate,
            offline_compression: a.offline_compression.as_str().to_string(),
            offline_top_k: a.offline_top_k,
            loss_type: a.loss_type.as_str().to_string(),
            skew_alpha: a.skew_alpha,
            universal_logit_top_k: a.universal_logit_top_k,
            temperature: a.temperature,
            alpha: a.alpha,
            rationale: a.rationale,
            rationale_weight: a.rationale_weight,
            on_policy: a.on_policy,
            on_policy_max_tokens: a.on_policy_max_tokens,
            lora_r: a.lora_r,
            lora_alpha: a.lora_alpha,
            learning_rate: a.learning_rate,
            optimizer: a.optimizer.to_string(),
            batch_size: a.batch_size,
            epochs: a.epochs,
            max_seq_len: a.max_seq_len,
            seed: a.seed,
            checkpoint_every: a.checkpoint_every,
            text_column: a.text_column,
            text_columns: a.text_columns.map(|names| names.join(",")),
            column_separator: a.column_separator,
            prompt_column: a.prompt_column,
            response_column: a.response_column,
            log_metrics: a.log_metrics,
            resume: a.resume,
        }
    }
}
