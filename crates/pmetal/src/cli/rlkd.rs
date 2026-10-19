//! Clap argument struct for `pmetal rlkd`.

use clap::Args;
use pmetal_core::jobs::RlkdSpec;

/// RLKD: Reinforcement Learning with Knowledge Distillation.
///
/// Combines GRPO policy gradient optimization with knowledge distillation
/// from a teacher model in a single training loop.
///
/// Loss formula: L = (1 - alpha) * L_grpo + alpha * L_distill
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
    #[arg(short, long = "output", default_value = pmetal_core::defaults::RLKD_OUTPUT_DIR)]
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
    #[arg(long = "beta", default_value = "0.0")]
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

impl From<RlkdArgs> for RlkdSpec {
    fn from(a: RlkdArgs) -> Self {
        RlkdSpec {
            model: a.model,
            teacher_model: a.teacher_model,
            dataset: a.dataset,
            output_dir: a.output,
            distill_alpha: a.distill_alpha,
            final_alpha: a.final_alpha,
            anneal_alpha: a.anneal_alpha,
            distill_temperature: a.distill_temperature,
            num_generations: a.num_generations,
            beta: a.beta,
            learning_rate: a.learning_rate,
            epochs: a.epochs,
            lora_r: a.lora_r,
            lora_alpha: a.lora_alpha,
            max_seq_len: a.max_seq_len,
            max_completion_length: a.max_completion_length,
            seed: a.seed,
            reasoning_rewards: a.reasoning_rewards,
            prompt_column: a.prompt_column,
            response_column: a.response_column,
            log_metrics: a.log_metrics,
        }
    }
}
