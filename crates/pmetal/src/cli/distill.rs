//! Clap argument struct for `pmetal distill`.

use clap::Args;

/// Thin clap argument struct for `pmetal distill`.
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

    /// Output directory for distilled student
    #[arg(short, long = "output", default_value = "./output/distilled")]
    pub output: String,

    /// Distillation method (online, offline, progressive)
    #[arg(long = "method", default_value = "online")]
    pub method: String,

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
    #[arg(long = "offline-compression", default_value = "top_k")]
    pub offline_compression: String,

    /// Top-k width for newly created offline logit caches.
    #[arg(long = "offline-top-k", default_value_t = 128)]
    pub offline_top_k: usize,

    /// Loss type (kl_divergence, jensen_shannon, soft_cross_entropy, mse_loss)
    #[arg(long = "loss-type", default_value = "kl_divergence")]
    pub loss_type: String,

    /// Softmax temperature
    #[arg(long = "temperature", default_value = "2.0")]
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

    /// LoRA rank for student
    #[arg(long = "lora-r", default_value = "16")]
    pub lora_r: usize,

    /// LoRA alpha scaling factor
    #[arg(long = "lora-alpha", default_value = "32")]
    pub lora_alpha: f32,

    /// Learning rate
    #[arg(long = "learning-rate", default_value = "2e-5")]
    pub learning_rate: f32,

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

    /// Custom text column name in the dataset JSONL.
    #[arg(long = "text-column")]
    pub text_column: Option<String>,

    /// Comma-separated list of columns to concatenate as the text field.
    #[arg(long = "text-columns", value_delimiter = ',')]
    pub text_columns: Option<Vec<String>>,

    /// Separator used when joining multiple text columns.
    #[arg(long = "column-separator", default_value = "\n\n")]
    pub column_separator: String,

    /// Column name for the prompt portion (enables SFT label masking).
    #[arg(long = "prompt-column")]
    pub prompt_column: Option<String>,

    /// Column name for the response portion (enables SFT label masking).
    #[arg(long = "response-column")]
    pub response_column: Option<String>,

    /// Path to write JSONL metrics log (for TUI dashboard)
    #[arg(long = "log-metrics")]
    pub log_metrics: Option<String>,
}
