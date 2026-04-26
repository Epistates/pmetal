//! Clap argument struct for `pmetal grpo`.

use clap::Args;

/// Thin clap argument struct for `pmetal grpo`.
#[derive(Args, Debug)]
pub struct GrpoArgs {
    /// Model ID or path
    #[arg(short, long = "model")]
    pub model: String,

    /// Dataset path (JSONL with prompts)
    #[arg(short, long = "dataset")]
    pub dataset: String,

    /// Output directory
    #[arg(short, long = "output", default_value = "./output/grpo")]
    pub output: String,

    /// Number of generations per prompt (group size)
    #[arg(long = "num-generations", default_value = "8")]
    pub num_generations: usize,

    /// KL penalty coefficient (beta)
    #[arg(long = "beta", default_value = "0.001")]
    pub beta: f64,

    /// Learning rate
    #[arg(long = "learning-rate", default_value = "5e-6")]
    pub learning_rate: f64,

    /// Number of training epochs
    #[arg(long = "epochs", default_value = "1")]
    pub epochs: usize,

    /// LoRA rank for policy model
    #[arg(long = "lora-r", default_value = "16")]
    pub lora_r: usize,

    /// LoRA alpha scaling factor
    #[arg(long = "lora-alpha", default_value = "32")]
    pub lora_alpha: f32,

    /// Maximum sequence length for generations
    #[arg(long = "max-seq-len", default_value = "512")]
    pub max_seq_len: usize,

    /// Maximum completion length per generation
    #[arg(long = "max-completion-length", default_value = "512")]
    pub max_completion_length: usize,

    /// Random seed for reproducibility
    #[arg(long = "seed", default_value = "42")]
    pub seed: u64,

    /// Enable DAPO (Distribution-Aware Policy Optimization)
    #[arg(long = "dapo")]
    pub dapo: bool,

    /// Use reasoning-aware rewards (e.g., length, formatting)
    #[arg(long = "reasoning-rewards")]
    pub reasoning_rewards: bool,

    /// Disable Metal FlashAttention
    #[arg(long = "no-flash-attention")]
    pub no_flash_attention: bool,

    /// Enable VLM (Vision-Language Model) mode for GRPO with image inputs.
    #[arg(long = "vlm")]
    pub vlm: bool,

    /// Maximum image size (pixels per side) for VLM preprocessing.
    #[arg(long = "max-image-size", default_value = "336")]
    pub max_image_size: usize,

    /// Path to a pretrained ML reward model for scoring completions.
    #[arg(long = "reward-model")]
    pub reward_model: Option<String>,

    /// Maximum input sequence length for the ML reward model (tokens).
    #[arg(long = "reward-model-max-length", default_value = "2048")]
    pub reward_model_max_length: usize,

    /// Weight for the ML reward model in the combined reward.
    #[arg(long = "reward-model-weight", default_value = "1.0")]
    pub reward_model_weight: f64,

    /// Chat template for the reward model (optional).
    #[arg(long = "reward-model-template")]
    pub reward_model_template: Option<String>,

    /// Enable speculative decoding for faster rollout generation.
    #[arg(long = "speculative")]
    pub speculative: bool,

    /// Number of draft tokens per speculative decode step (default: 3).
    #[arg(long = "speculative-draft-tokens", default_value = "3")]
    pub speculative_draft_tokens: usize,

    /// Enable pipelined (asynchronous) reward scoring.
    #[arg(long = "async-rewards")]
    pub async_rewards: bool,

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

    /// KV cache quantization bits for GRPO rollout generation (2, 4, or 8).
    #[arg(long = "grpo-kv-bits")]
    pub grpo_kv_bits: Option<u8>,

    /// Path to write JSONL metrics log (for TUI dashboard)
    #[arg(long = "log-metrics")]
    pub log_metrics: Option<String>,
}
