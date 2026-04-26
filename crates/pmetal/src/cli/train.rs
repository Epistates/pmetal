//! Clap argument struct for `pmetal train`.
//!
//! Field `#[arg(long = "...")]` strings MUST match the `argv = "..."` attribute
//! in `TrainSpec` (pmetal-core/src/jobs/train.rs).

use clap::Args;

/// Thin clap argument struct for `pmetal train`.
#[derive(Args, Debug)]
pub struct TrainArgs {
    /// Path to training configuration file (YAML)
    #[arg(short, long)]
    pub config: Option<String>,

    /// Model ID (HuggingFace or local path)
    #[arg(short, long = "model")]
    pub model: Option<String>,

    /// Dataset path (JSONL file)
    #[arg(short, long = "dataset")]
    pub dataset: Option<String>,

    /// Evaluation dataset path (optional JSONL file)
    #[arg(long = "eval-dataset")]
    pub eval_dataset: Option<String>,

    /// Output directory
    #[arg(short, long = "output", default_value = "./output")]
    pub output: String,

    /// LoRA rank
    #[arg(long = "lora-r", default_value = "16")]
    pub lora_r: usize,

    /// LoRA alpha (scaling factor). Recommended: 2x rank.
    #[arg(long = "lora-alpha", default_value = "32")]
    pub lora_alpha: f32,

    /// Learning rate. Recommended: 2e-4 for most tasks.
    #[arg(long = "learning-rate", default_value = "2e-4")]
    pub learning_rate: f64,

    /// Batch size
    #[arg(long = "batch-size", default_value = "1")]
    pub batch_size: usize,

    /// Number of epochs
    #[arg(long = "epochs", default_value = "1")]
    pub epochs: usize,

    /// Maximum sequence length (0 to auto-detect from model config)
    #[arg(long = "max-seq-len", default_value = "0")]
    pub max_seq_len: usize,

    /// Gradient accumulation steps
    #[arg(long = "gradient-accumulation-steps", default_value = "4")]
    pub gradient_accumulation_steps: usize,

    /// Disable Metal FlashAttention (enabled by default for O(n) memory)
    #[arg(long = "no-flash-attention")]
    pub no_flash_attention: bool,

    /// Maximum gradient norm for clipping (0 to disable)
    #[arg(long = "max-grad-norm", default_value = "1.0")]
    pub max_grad_norm: f64,

    /// Resume from checkpoint
    #[arg(long = "resume")]
    pub resume: bool,

    /// Quantization method for QLoRA (none, nf4, fp4, int8)
    #[arg(long = "quantization", value_enum, default_value = "none")]
    pub quantization: crate::QuantizationMethod,

    /// Block size for quantization (default: 64)
    #[arg(long = "quant-block-size", default_value = "64")]
    pub quant_block_size: usize,

    /// Enable double quantization for absmax values
    #[arg(long = "double-quant")]
    pub double_quant: bool,

    /// Disable fused training step
    #[arg(long = "no-fused")]
    pub no_fused: bool,

    /// Disable Metal fused optimizer
    #[arg(long = "no-metal-fused-optimizer")]
    pub no_metal_fused_optimizer: bool,

    /// Disable sequence packing
    #[arg(long = "no-sequence-packing")]
    pub no_sequence_packing: bool,

    /// Override the maximum sequence length for packing.
    #[arg(long = "pack-max-seq-len")]
    pub pack_max_seq_len: Option<usize>,

    /// Disable the experimental compiled-training dispatch.
    #[arg(long = "no-jit-compilation")]
    pub no_jit_compilation: bool,

    /// Disable gradient checkpointing
    #[arg(long = "no-gradient-checkpointing")]
    pub no_gradient_checkpointing: bool,

    /// Number of layers per checkpoint block (default: 4).
    #[arg(long = "gradient-checkpointing-layers", default_value = "4")]
    pub gradient_checkpointing_layers: usize,

    /// Path to log training metrics as JSONL.
    #[arg(long = "log-metrics")]
    pub log_metrics: Option<String>,

    /// Separate learning rate for embedding layers.
    #[arg(long = "embedding-lr")]
    pub embedding_lr: Option<f32>,

    /// Loss scaling factor for ANE training (default: 1.0).
    #[arg(long = "loss-scale", default_value = "1.0")]
    pub loss_scale: f32,

    /// Number of linear warmup steps.
    #[arg(long = "warmup-steps", default_value = "0")]
    pub warmup_steps: usize,

    /// Learning rate schedule.
    #[arg(long = "lr-schedule", default_value = "cosine")]
    pub lr_schedule: String,

    /// AdamW weight decay coefficient.
    #[arg(long = "weight-decay", default_value = "0.01")]
    pub weight_decay: f64,

    /// Random seed.
    #[arg(long = "seed", default_value = "42")]
    pub seed: u64,

    /// Use Cut Cross-Entropy for memory-efficient loss computation.
    #[arg(long = "cut-cross-entropy")]
    pub cut_cross_entropy: bool,

    /// Disable automatic adaptive LR.
    #[arg(long = "no-adaptive-lr")]
    pub no_adaptive_lr: bool,

    /// Custom JSONL column containing the training text.
    #[arg(long = "text-column")]
    pub text_column: Option<String>,

    /// Multiple JSONL columns to concatenate as the training text.
    #[arg(long = "text-columns", value_delimiter = ',')]
    pub text_columns: Option<Vec<String>>,

    /// Separator inserted between columns when using --text-columns.
    #[arg(long = "column-separator", default_value = "\n\n")]
    pub column_separator: String,

    /// Custom JSONL column containing the prompt portion (loss-masked).
    #[arg(long = "prompt-column")]
    pub prompt_column: Option<String>,

    /// Custom JSONL column containing the response portion.
    #[arg(long = "response-column")]
    pub response_column: Option<String>,

    /// Enable ANE (Apple Neural Engine) for training (experimental).
    #[cfg(feature = "ane")]
    #[arg(long = "ane")]
    pub ane: bool,

    /// Distributed training: comma-separated peer addresses (ip:port).
    #[cfg(feature = "distributed")]
    #[arg(long = "distributed-peers", value_delimiter = ',')]
    pub distributed_peers: Option<Vec<String>>,

    /// Distributed training: enable automatic mDNS peer discovery.
    #[cfg(feature = "distributed")]
    #[arg(long = "distributed-auto")]
    pub distributed_auto: bool,

    /// Gradient compression strategy for distributed training.
    #[cfg(feature = "distributed")]
    #[arg(long = "compression-strategy", default_value = "none")]
    pub compression_strategy: Option<String>,
}
