//! Clap argument struct for `pmetal grpo`.

use clap::Args;
use pmetal_core::jobs::GrpoSpec;

/// Group Relative Policy Optimization (GRPO) for reasoning models.
#[derive(Args, Debug)]
pub struct GrpoArgs {
    /// Model ID or path
    #[arg(short, long = "model")]
    pub model: String,

    /// Dataset path (JSONL with prompts)
    #[arg(short, long = "dataset")]
    pub dataset: String,

    /// Output directory
    #[arg(short, long = "output", default_value = pmetal_core::defaults::GRPO_OUTPUT_DIR)]
    pub output: String,

    /// Number of generations per prompt (group size)
    #[arg(long = "num-generations", default_value = "8")]
    pub num_generations: usize,

    /// KL penalty coefficient (beta)
    #[arg(long = "beta", default_value = "0.0")]
    pub beta: f64,

    /// Learning rate
    #[arg(long = "learning-rate", default_value = "5e-6")]
    pub learning_rate: f64,

    /// Optimizer: adamw, sgd, lion, adafactor, muon, adamw_schedule_free or
    /// adamw_8bit. Lion wants a 3-10x smaller learning rate and 3-10x larger
    /// weight decay than AdamW; schedule-free AdamW wants a constant rate.
    #[arg(long = "optimizer", default_value = "adamw")]
    pub optimizer: pmetal_core::OptimizerType,

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

    /// Save LoRA checkpoint every N policy updates (0 disables interval checkpoints).
    #[arg(long = "checkpoint-every", default_value = "50")]
    pub checkpoint_every: usize,

    /// Resume LoRA weights from the latest checkpoint in the output directory.
    #[arg(long = "resume")]
    pub resume: bool,

    /// The DAPO recipe (Decoupled Clip and Dynamic sAmpling Policy
    /// Optimization): token-level loss, no KL term, clip-higher (0.2, 0.28),
    /// dynamic sampling, an overlong penalty and at least 16 completions.
    #[arg(long = "dapo")]
    pub dapo: bool,

    /// How token losses combine: dapo (token-level, the default), grpo (each
    /// completion averaged over its length), dr_grpo (a constant normalizer
    /// and no std scaling of advantages) or gspo (a sequence-level policy
    /// ratio with clip range 3e-4/4e-4; needs --num-iterations above 1).
    #[arg(long = "loss-type", default_value = "dapo")]
    pub loss_type: String,

    /// Optimizer updates per generation batch. Above 1 the policy ratio
    /// moves off 1 and clipping engages.
    #[arg(long = "num-iterations", default_value = "1")]
    pub num_iterations: usize,

    /// Use reasoning-aware rewards (e.g., length, formatting)
    #[arg(long = "reasoning-rewards")]
    pub reasoning_rewards: bool,

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

    /// Enable pipelined (asynchronous) reward scoring.
    #[arg(long = "async-rewards")]
    pub async_rewards: bool,

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

impl From<GrpoArgs> for GrpoSpec {
    fn from(a: GrpoArgs) -> Self {
        GrpoSpec {
            model: a.model,
            dataset: a.dataset,
            output_dir: a.output,
            num_generations: a.num_generations,
            beta: a.beta,
            learning_rate: a.learning_rate,
            optimizer: a.optimizer.to_string(),
            epochs: a.epochs,
            lora_r: a.lora_r,
            lora_alpha: a.lora_alpha,
            max_seq_len: a.max_seq_len,
            max_completion_length: a.max_completion_length,
            seed: a.seed,
            checkpoint_every: a.checkpoint_every,
            dapo: a.dapo,
            loss_type: a.loss_type,
            num_iterations: a.num_iterations,
            reasoning_rewards: a.reasoning_rewards,
            vlm: a.vlm,
            max_image_size: a.max_image_size,
            reward_model: a.reward_model,
            reward_model_max_length: a.reward_model_max_length,
            reward_model_weight: a.reward_model_weight,
            reward_model_template: a.reward_model_template,
            async_rewards: a.async_rewards,
            grpo_kv_bits: a.grpo_kv_bits,
            prompt_column: a.prompt_column,
            response_column: a.response_column,
            log_metrics: a.log_metrics,
            resume: a.resume,
        }
    }
}
