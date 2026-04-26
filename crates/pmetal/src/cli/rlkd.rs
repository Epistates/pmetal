//! Clap argument struct for `pmetal rlkd`.

use clap::Args;

/// Thin clap argument struct for `pmetal rlkd`.
#[derive(Args, Debug)]
pub struct RlkdArgs {
    /// Policy (student) model ID or local path.
    #[arg(short, long = "model")]
    pub model: String,

    /// Teacher model ID or local path (frozen, provides soft targets).
    #[arg(long = "teacher-model")]
    pub teacher_model: String,

    /// Dataset path (JSONL with prompts).
    #[arg(short, long = "dataset")]
    pub dataset: String,

    /// Output directory for LoRA adapter weights.
    #[arg(short, long = "output", default_value = "./output/rlkd")]
    pub output: String,

    /// Distillation blend factor: 0.0 = pure RL, 1.0 = pure distillation.
    #[arg(long = "distill-alpha", default_value = "0.3")]
    pub distill_alpha: f32,

    /// Final alpha value when annealing (default: 0.05 = mostly RL by end).
    #[arg(long = "final-alpha", default_value = "0.05")]
    pub final_alpha: f32,

    /// Linearly anneal alpha from `--distill-alpha` toward `--final-alpha`.
    #[arg(long = "anneal-alpha")]
    pub anneal_alpha: bool,

    /// Temperature for distillation soft targets (default: 2.0).
    #[arg(long = "distill-temperature", default_value = "2.0")]
    pub distill_temperature: f32,

    /// Number of completions to generate per prompt (GRPO group size).
    #[arg(long = "num-generations", default_value = "8")]
    pub num_generations: usize,

    /// KL penalty coefficient (beta) for GRPO reference model regularization.
    #[arg(long = "beta", default_value = "0.001")]
    pub beta: f64,

    /// Learning rate.
    #[arg(long = "learning-rate", default_value = "5e-6")]
    pub learning_rate: f64,

    /// Number of training epochs.
    #[arg(long = "epochs", default_value = "1")]
    pub epochs: usize,

    /// LoRA rank for the policy model.
    #[arg(long = "lora-r", default_value = "16")]
    pub lora_r: usize,

    /// LoRA alpha scaling factor.
    #[arg(long = "lora-alpha", default_value = "32")]
    pub lora_alpha: f32,

    /// Maximum sequence length (prompt + completion).
    #[arg(long = "max-seq-len", default_value = "512")]
    pub max_seq_len: usize,

    /// Maximum completion length per generation.
    #[arg(long = "max-completion-length", default_value = "512")]
    pub max_completion_length: usize,

    /// Random seed for reproducibility.
    #[arg(long = "seed", default_value = "42")]
    pub seed: u64,

    /// Use reasoning-aware rewards (format + length signals).
    #[arg(long = "reasoning-rewards")]
    pub reasoning_rewards: bool,

    /// Disable Metal FlashAttention.
    #[arg(long = "no-flash-attention")]
    pub no_flash_attention: bool,

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

    /// Path to write JSONL metrics log (for TUI dashboard).
    #[arg(long = "log-metrics")]
    pub log_metrics: Option<String>,
}
