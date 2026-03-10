//! PMetal CLI - LLM fine-tuning for Apple Silicon.

#![allow(clippy::too_many_arguments)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::needless_borrows_for_generic_args)]

mod dashboard;
#[cfg(feature = "dashboard")]
mod tui;

use std::path::{Component, Path, PathBuf};

use clap::{Parser, Subcommand, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use mlx_rs::builder::Builder;
use pmetal_core::{DatasetConfig, LoraConfig, ModelConfig, TrainingConfig};
use pmetal_data::{
    DataLoaderConfig, DatasetFormat, DatasetSource, Tokenizer, TrainingDataset,
    resolve_dataset_source,
};
use pmetal_lora::{
    DynamicLoraModel, LlamaLoraForCausalLM, LlamaQloraForCausalLM, QLoraConfig, TrainableModel,
};
use pmetal_mlx::quantization::QuantScheme;
use pmetal_models::architectures::llama::LlamaConfig;
use pmetal_models::ollama::{ModelfileBuilder, templates as ollama_templates};
use pmetal_models::{DynamicModel, WeightFormat};
use pmetal_trainer::{
    CheckpointManager, GrpoConfig, GrpoTrainer, MetricsJsonCallback, TrainingLoop,
    TrainingLoopConfig,
};
use serde::{Deserialize, Serialize};

/// Combined configuration for training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullTrainingConfig {
    /// Model configuration.
    #[serde(default)]
    pub model: ModelConfig,

    /// LoRA configuration.
    #[serde(default)]
    pub lora: LoraConfig,

    /// Training hyperparameters.
    #[serde(default)]
    pub training: TrainingConfig,

    /// Dataset configuration.
    #[serde(default)]
    pub dataset: DatasetConfig,
}

impl Default for FullTrainingConfig {
    fn default() -> Self {
        Self {
            model: ModelConfig::default(),
            lora: LoraConfig::default(),
            training: TrainingConfig::default(),
            dataset: DatasetConfig::default(),
        }
    }
}

/// Quantization method for QLoRA.
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
pub enum QuantizationMethod {
    /// No quantization (standard LoRA)
    #[default]
    None,
    /// NF4 (Normal Float 4-bit) - recommended, optimal for normally distributed weights
    Nf4,
    /// FP4 (Float Point 4-bit)
    Fp4,
    /// INT8 (8-bit integer)
    Int8,
}

#[derive(Parser)]
#[command(name = "pmetal")]
#[command(author, version, about = "LLM fine-tuning optimized for Apple Silicon", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Fine-tune a model using LoRA/QLoRA
    Train {
        /// Path to training configuration file (YAML)
        #[arg(short, long)]
        config: Option<String>,

        /// Model ID (HuggingFace or local path)
        #[arg(short, long)]
        model: Option<String>,

        /// Dataset path (JSONL file)
        #[arg(short, long)]
        dataset: Option<String>,

        /// Evaluation dataset path (optional JSONL file)
        #[arg(long)]
        eval_dataset: Option<String>,

        /// Output directory
        #[arg(short, long, default_value = "./output")]
        output: String,

        /// LoRA rank
        #[arg(long, default_value = "16")]
        lora_r: usize,

        /// LoRA alpha (scaling factor). Unsloth recommends 2x rank.
        #[arg(long, default_value = "32")]
        lora_alpha: f32,

        /// Learning rate. Unsloth recommends 2e-4 for most tasks.
        #[arg(long, default_value = "2e-4")]
        learning_rate: f64,

        /// Batch size
        #[arg(long, default_value = "1")]
        batch_size: usize,

        /// Number of epochs
        #[arg(long, default_value = "1")]
        epochs: usize,

        /// Maximum sequence length (0 to auto-detect from model config)
        #[arg(long, default_value = "0")]
        max_seq_len: usize,

        /// Gradient accumulation steps
        #[arg(long, default_value = "4")]
        gradient_accumulation_steps: usize,

        /// Disable Metal FlashAttention (enabled by default for O(n) memory)
        #[arg(long)]
        no_flash_attention: bool,

        /// Maximum gradient norm for clipping (0 to disable)
        #[arg(long, default_value = "1.0")]
        max_grad_norm: f64,

        /// Resume from checkpoint
        #[arg(long)]
        resume: bool,

        /// Quantization method for QLoRA (none, nf4, fp4, int8)
        #[arg(long, value_enum, default_value = "none")]
        quantization: QuantizationMethod,

        /// Block size for quantization (default: 64)
        #[arg(long, default_value = "64")]
        quant_block_size: usize,

        /// Enable double quantization for absmax values
        #[arg(long)]
        double_quant: bool,

        /// Disable fused training step (enabled by default when gradient_accumulation_steps=1)
        #[arg(long)]
        no_fused: bool,

        /// Disable Metal fused optimizer (enabled by default for ~40% throughput improvement)
        #[arg(long)]
        no_metal_fused_optimizer: bool,

        /// Disable sequence packing (enabled by default for 2-5x throughput)
        #[arg(long)]
        no_sequence_packing: bool,

        /// Disable JIT compilation (enabled by default for up to 50% throughput improvement)
        #[arg(long)]
        no_jit_compilation: bool,

        /// Disable gradient checkpointing (enabled by default for memory efficiency)
        #[arg(long)]
        no_gradient_checkpointing: bool,

        /// Number of layers per checkpoint block (default: 4).
        /// Lower = more memory savings but slower.
        #[arg(long, default_value = "4")]
        gradient_checkpointing_layers: usize,

        /// Path to log training metrics as JSONL (Wandb-compatible).
        /// Metrics can be imported to Wandb: `wandb sync path/to/metrics.jsonl`
        #[arg(long)]
        log_metrics: Option<String>,

        /// Separate learning rate for embedding layers.
        /// Unsloth recommends 5e-5 for embeddings vs 2e-4 for LoRA params.
        /// Improves training stability for large vocabulary models.
        #[arg(long)]
        embedding_lr: Option<f32>,

        /// Disable ANE (Apple Neural Engine) for training, falling back to GPU/MLX.
        #[cfg(feature = "ane")]
        #[arg(long)]
        no_ane: bool,
    },

    /// Run inference with a model
    Infer {
        /// Model ID or path
        #[arg(short, long)]
        model: String,

        /// LoRA adapter path (optional)
        #[arg(long)]
        lora: Option<String>,

        /// Input prompt
        #[arg(short, long)]
        prompt: String,

        /// Maximum tokens to generate
        #[arg(long, default_value = "256")]
        max_tokens: usize,

        /// Temperature for sampling (0 = greedy). Defaults to model's generation_config.json
        #[arg(long)]
        temperature: Option<f32>,

        /// Top-k sampling (0 = disabled). Defaults to model's generation_config.json
        #[arg(long)]
        top_k: Option<usize>,

        /// Top-p nucleus sampling (0.0-1.0). Defaults to model's generation_config.json
        #[arg(long)]
        top_p: Option<f32>,

        /// Min-p dynamic sampling (0.0 = disabled). Defaults to model's generation_config.json
        #[arg(long)]
        min_p: Option<f32>,

        /// Repetition penalty applied to prompt + output (1.0 = disabled, 1.0-1.2 typical)
        #[arg(long)]
        repetition_penalty: Option<f32>,

        /// Frequency penalty proportional to token count (0.0 = disabled, 0.0-2.0 typical)
        #[arg(long)]
        frequency_penalty: Option<f32>,

        /// Presence penalty for any appeared token (0.0 = disabled, Qwen3 recommends 0-2)
        #[arg(long)]
        presence_penalty: Option<f32>,

        /// Random seed for reproducible generation
        #[arg(long)]
        seed: Option<u64>,

        /// Apply chat template (auto-detected from tokenizer)
        #[arg(long)]
        chat: bool,

        /// System message for chat mode
        #[arg(long)]
        system: Option<String>,

        /// Disable thinking mode for models that support it (e.g., Qwen3)
        /// By default, the model decides when to use thinking based on query complexity
        #[arg(long)]
        no_thinking: bool,

        /// Use fused Metal sampling kernel for better battery performance
        /// (bypasses mlx-rs sampling, uses single GPU kernel launch)
        #[arg(long)]
        metal_sampler: bool,

        /// Use JIT-compiled sampling for better performance
        /// (matches mlx_lm's @mx.compile approach for kernel fusion)
        #[arg(long)]
        compiled: bool,

        /// Use dedicated GPU stream for generation (like mlx_lm's generation_stream)
        /// (may improve pipelining and reduce scheduling overhead)
        #[arg(long)]
        stream: bool,

        /// Use minimal async generation (for performance debugging)
        #[arg(long)]
        minimal: bool,

        /// Show thinking content in output (if model generates it)
        #[arg(long)]
        show_thinking: bool,

        /// Use FP8 quantization for weights (~2x memory reduction).
        /// Quantizes model weights to 8-bit floating point (E4M3 format)
        /// for memory-efficient inference on Apple Silicon.
        #[arg(long)]
        fp8: bool,

        /// Disable ANE (Apple Neural Engine) for inference, falling back to GPU/Metal.
        #[cfg(feature = "ane")]
        #[arg(long)]
        no_ane: bool,

        /// Maximum ANE kernel sequence length (power-of-2 bucket cap).
        /// ANE kernels are compiled for a fixed spatial dimension — larger values
        /// allow longer prompts to be processed on ANE but may fail to compile
        /// for models with many attention heads. Default: 1024.
        #[cfg(feature = "ane")]
        #[arg(long, default_value = "1024")]
        ane_max_seq_len: usize,
    },

    /// Download a model from HuggingFace
    Download {
        /// Model ID
        model: String,

        /// Specific revision
        #[arg(long)]
        revision: Option<String>,
    },

    /// Show memory usage and available capacity
    Memory,

    /// Benchmark FFI overhead (for performance analysis)
    BenchFfi,

    /// Benchmark generation loop timing (detailed profiling)
    BenchGen {
        /// Model to benchmark
        #[arg(short, long, default_value = "unsloth/Qwen3-0.6B")]
        model: String,
    },

    /// Benchmark training performance
    Bench {
        /// Model to benchmark
        #[arg(short, long, default_value = "meta-llama/Llama-3.2-1B")]
        model: String,

        /// Batch size
        #[arg(short, long, default_value = "1")]
        batch_size: usize,

        /// Sequence length
        #[arg(short, long, default_value = "512")]
        seq_len: usize,
    },

    /// Generate a sample configuration file
    Init {
        /// Output path for the config file
        #[arg(short, long, default_value = "config.yaml")]
        output: String,
    },

    /// Export trained model for Ollama
    Ollama {
        #[command(subcommand)]
        action: OllamaAction,
    },

    /// Fuse LoRA adapter weights into a base model and save as a complete model
    Fuse {
        /// Base model ID or path
        #[arg(short, long)]
        model: String,

        /// LoRA adapter path (directory containing lora_weights.safetensors, or the file itself)
        #[arg(short, long)]
        lora: String,

        /// Output directory for the fused model
        #[arg(short, long)]
        output: String,

        /// LoRA scaling alpha (default: auto-detect from adapter)
        #[arg(long)]
        alpha: Option<f32>,

        /// LoRA rank (default: auto-detect from adapter)
        #[arg(long)]
        rank: Option<usize>,
    },

    /// Quantize a model to GGUF format (supports Dynamic 2.0)
    Quantize {
        /// Source model path (Safetensors/HF)
        #[arg(short, long)]
        model: String,

        /// Output GGUF file path
        #[arg(short, long)]
        output: String,

        /// Path to Importance Matrix (imatrix.dat) for dynamic quantization
        #[arg(long)]
        imatrix: Option<String>,

        /// Quantization method (e.g., q4_k_m, q8_0) or "dynamic"
        #[arg(long, default_value = "dynamic")]
        method: String,

        /// LoRA adapter to fuse before quantizing (optional)
        #[arg(long)]
        lora: Option<String>,
    },

    /// Knowledge Distillation from teacher to student
    Distill {
        /// Teacher model ID or path
        #[arg(short, long)]
        teacher: String,

        /// Student model ID or path
        #[arg(short, long)]
        student: String,

        /// Dataset path (JSONL file)
        #[arg(short, long)]
        dataset: String,

        /// Output directory for distilled student
        #[arg(short, long, default_value = "./output/distilled")]
        output: String,

        /// Distillation method (online, offline, progressive)
        #[arg(long, default_value = "online")]
        method: String,

        /// Loss type (kl_divergence, jensen_shannon, soft_cross_entropy, mse_loss)
        #[arg(long, default_value = "kl_divergence")]
        loss_type: String,

        /// Softmax temperature
        #[arg(long, default_value = "2.0")]
        temperature: f32,

        /// Alpha for blending hard/soft targets (0.0 to 1.0)
        #[arg(long, default_value = "0.5")]
        alpha: f32,

        /// Use reasoning-aware (rationale) distillation
        #[arg(long)]
        rationale: bool,

        /// Weight for reasoning tokens (if rationale is enabled)
        #[arg(long, default_value = "1.0")]
        rationale_weight: f32,

        /// LoRA rank for student
        #[arg(long, default_value = "16")]
        lora_r: usize,

        /// Learning rate
        #[arg(long, default_value = "2e-5")]
        learning_rate: f32,

        /// Batch size
        #[arg(long, default_value = "1")]
        batch_size: usize,

        /// Number of epochs
        #[arg(long, default_value = "1")]
        epochs: usize,

        /// Maximum sequence length
        #[arg(long, default_value = "1024")]
        max_seq_len: usize,
    },

    /// Group Relative Policy Optimization (GRPO) for reasoning models
    Grpo {
        /// Model ID or path
        #[arg(short, long)]
        model: String,

        /// Dataset path (JSONL with prompts)
        #[arg(short, long)]
        dataset: String,

        /// Output directory
        #[arg(short, long, default_value = "./output/grpo")]
        output: String,

        /// Number of generations per prompt (group size)
        #[arg(long, default_value = "8")]
        num_generations: usize,

        /// KL penalty coefficient (beta)
        #[arg(long, default_value = "0.001")]
        beta: f64,

        /// Learning rate
        #[arg(long, default_value = "5e-6")]
        learning_rate: f64,

        /// Maximum sequence length for generations
        #[arg(long, default_value = "512")]
        max_seq_len: usize,

        /// Enable DAPO (Distribution-Aware Policy Optimization)
        #[arg(long)]
        dapo: bool,

        /// Use reasoning-aware rewards (e.g., length, formatting)
        #[arg(long)]
        reasoning_rewards: bool,

        /// Disable Metal FlashAttention
        #[arg(long)]
        no_flash_attention: bool,
    },

    /// Dataset utilities for preparing and analyzing training data
    Dataset {
        #[command(subcommand)]
        action: DatasetAction,
    },

    /// Real-time training dashboard (loss curves, ANE utilization, timing)
    Dashboard {
        /// Path to training metrics JSONL file to visualize
        #[arg(short, long)]
        metrics_file: Option<String>,
    },

    /// Full TUI control center (device, models, datasets, training, inference)
    #[cfg(feature = "dashboard")]
    Tui {
        /// Path to training metrics JSONL file to visualize in the dashboard tab
        #[arg(short, long)]
        metrics_file: Option<String>,
    },
}

/// Dataset subcommands for data preparation.
#[derive(Subcommand)]
enum DatasetAction {
    /// Analyze a dataset and show statistics
    Analyze {
        /// Path to dataset file (JSONL)
        #[arg(short, long)]
        path: String,

        /// Model ID for tokenization (required for accurate token counts)
        #[arg(short, long)]
        model: Option<String>,

        /// Show detailed per-sample statistics
        #[arg(long)]
        detailed: bool,
    },

    /// Download a dataset from HuggingFace Hub
    Download {
        /// Dataset ID (e.g., "tatsu-lab/alpaca", "TeichAI/gemini-3-pro-preview-high-reasoning-1000x")
        dataset_id: String,

        /// Dataset split to download (e.g., "train", "test")
        #[arg(long, default_value = "train")]
        split: String,

        /// Output path for converted JSONL file
        #[arg(short, long)]
        output: Option<String>,

        /// Specific revision/branch
        #[arg(long)]
        revision: Option<String>,
    },

    /// Convert a dataset to pmetal-compatible JSONL format
    Convert {
        /// Input file path (Parquet, JSON, JSONL, CSV)
        #[arg(short, long)]
        input: String,

        /// Output file path (JSONL)
        #[arg(short, long)]
        output: String,

        /// Input format (auto-detected if not specified)
        #[arg(long, value_enum)]
        format: Option<InputFormat>,

        /// Column mapping for non-standard formats (e.g., "text=content,prompt=instruction")
        #[arg(long)]
        columns: Option<String>,

        /// Shuffle the output data
        #[arg(long)]
        shuffle: bool,

        /// Random seed for shuffling
        #[arg(long, default_value = "42")]
        seed: u64,
    },

    /// Validate a dataset for training
    Validate {
        /// Path to dataset file (JSONL)
        #[arg(short, long)]
        path: String,

        /// Model ID for tokenization validation
        #[arg(short, long)]
        model: Option<String>,

        /// Maximum sequence length to check against
        #[arg(long, default_value = "2048")]
        max_seq_len: usize,
    },

    /// Preview first N samples from a HuggingFace dataset
    Preview {
        /// Dataset ID (e.g., "tatsu-lab/alpaca")
        dataset_id: String,

        /// Dataset split (e.g., "train", "test")
        #[arg(long, default_value = "train")]
        split: String,

        /// Number of samples to preview
        #[arg(short, long, default_value = "5")]
        num: usize,
    },

    /// Filter a dataset by various criteria
    Filter {
        /// Input dataset path (JSONL)
        #[arg(short, long)]
        input: String,

        /// Output dataset path (JSONL)
        #[arg(short, long)]
        output: String,

        /// Model ID for token-based filtering
        #[arg(short, long)]
        model: Option<String>,

        /// Minimum token count (requires --model)
        #[arg(long)]
        min_tokens: Option<usize>,

        /// Maximum token count (requires --model)
        #[arg(long)]
        max_tokens: Option<usize>,

        /// Remove duplicate samples (exact match)
        #[arg(long)]
        dedup: bool,

        /// Regex pattern to match (keeps matching samples)
        #[arg(long)]
        pattern: Option<String>,

        /// Invert pattern matching (keeps non-matching samples)
        #[arg(long)]
        invert: bool,

        /// Require all conversation turns (filters incomplete conversations)
        #[arg(long)]
        complete_only: bool,
    },

    /// Split a dataset into train/validation/test sets
    Split {
        /// Input dataset path (JSONL)
        #[arg(short, long)]
        input: String,

        /// Output directory for split files
        #[arg(short, long)]
        output_dir: String,

        /// Validation set ratio (0.0 to 1.0)
        #[arg(long, default_value = "0.1")]
        val_ratio: f64,

        /// Test set ratio (0.0 to 1.0)
        #[arg(long, default_value = "0.0")]
        test_ratio: f64,

        /// Random seed for splitting
        #[arg(long, default_value = "42")]
        seed: u64,

        /// Stratify by a field (e.g., "category", "difficulty")
        #[arg(long)]
        stratify: Option<String>,
    },

    /// Merge multiple datasets into one
    Merge {
        /// Input dataset paths (JSONL)
        #[arg(short, long, num_args = 1..)]
        inputs: Vec<String>,

        /// Output dataset path (JSONL)
        #[arg(short, long)]
        output: String,

        /// Shuffle after merging
        #[arg(long)]
        shuffle: bool,

        /// Random seed for shuffling
        #[arg(long, default_value = "42")]
        seed: u64,

        /// Interleave samples from each dataset (alternating)
        #[arg(long)]
        interleave: bool,

        /// Weights for each input dataset (comma-separated, e.g., "1.0,2.0,0.5")
        #[arg(long)]
        weights: Option<String>,
    },

    /// Take a random sample from a dataset
    Sample {
        /// Input dataset path (JSONL)
        #[arg(short, long)]
        input: String,

        /// Output dataset path (JSONL)
        #[arg(short, long)]
        output: String,

        /// Number of samples to take
        #[arg(short, long)]
        num: usize,

        /// Random seed
        #[arg(long, default_value = "42")]
        seed: u64,
    },

    /// Apply a chat template to format conversations
    Template {
        /// Input dataset path (JSONL with conversations)
        #[arg(short, long)]
        input: String,

        /// Output dataset path (JSONL with formatted text)
        #[arg(short, long)]
        output: String,

        /// Chat template to use
        #[arg(long, value_enum, default_value = "chatml")]
        template: ChatTemplate,

        /// Custom system message (overrides default)
        #[arg(long)]
        system: Option<String>,

        /// Model ID for tokenizer-based template (uses tokenizer's chat_template)
        #[arg(short, long)]
        model: Option<String>,

        /// Add generation prompt marker at end
        #[arg(long)]
        add_generation_prompt: bool,

        /// Only mask prompt tokens in labels (for SFT)
        #[arg(long)]
        mask_prompt: bool,
    },

    /// Prepare a dataset for training (full pipeline)
    Prepare {
        /// HuggingFace dataset ID or local path
        dataset: String,

        /// Output directory
        #[arg(short, long)]
        output_dir: String,

        /// Model ID for tokenization
        #[arg(short, long)]
        model: String,

        /// Chat template to apply
        #[arg(long, value_enum, default_value = "chatml")]
        template: ChatTemplate,

        /// Maximum sequence length (filters longer samples)
        #[arg(long, default_value = "2048")]
        max_seq_len: usize,

        /// Validation split ratio
        #[arg(long, default_value = "0.05")]
        val_ratio: f64,

        /// Random seed
        #[arg(long, default_value = "42")]
        seed: u64,

        /// Skip deduplication
        #[arg(long)]
        no_dedup: bool,
    },

    /// Show supported formats and templates
    Formats,
}

/// Chat template formats for conversation formatting.
#[derive(Debug, Clone, Copy, ValueEnum)]
enum ChatTemplate {
    /// ChatML format: <|im_start|>role\ncontent<|im_end|>
    Chatml,
    /// Llama 3 format: <|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>
    Llama3,
    /// Llama 2 format: [INST] user [/INST] assistant </s>
    Llama2,
    /// Mistral/Zephyr format: <s>[INST] user [/INST] assistant</s>
    Mistral,
    /// Qwen format: <|im_start|>role\ncontent<|im_end|>
    Qwen,
    /// Phi format: <|user|>\ncontent<|end|>\n<|assistant|>\ncontent<|end|>
    Phi,
    /// Gemma format: <start_of_turn>role\ncontent<end_of_turn>
    Gemma,
    /// Raw text (no template, just concatenate)
    Raw,
    /// Use tokenizer's built-in chat template
    Auto,
}

/// Input format for dataset conversion.
#[derive(Debug, Clone, Copy, ValueEnum)]
enum InputFormat {
    /// Parquet format
    Parquet,
    /// JSON array format
    Json,
    /// JSONL (JSON Lines) format
    Jsonl,
    /// CSV format
    Csv,
    /// ShareGPT conversation format
    ShareGpt,
    /// Alpaca instruction format
    Alpaca,
}

/// Ollama subcommands for model export and registration.
#[derive(Subcommand)]
enum OllamaAction {
    /// Generate a Modelfile for a trained model
    Modelfile {
        /// Base model (GGUF path or Ollama model name)
        #[arg(short, long)]
        base: String,

        /// LoRA adapter path (optional)
        #[arg(long)]
        lora: Option<String>,

        /// Output Modelfile path
        #[arg(short, long, default_value = "Modelfile")]
        output: String,

        /// System prompt
        #[arg(long)]
        system: Option<String>,

        /// Temperature (0.0-2.0)
        #[arg(long)]
        temperature: Option<f32>,

        /// Context window size
        #[arg(long)]
        num_ctx: Option<i32>,

        /// Top-k sampling
        #[arg(long)]
        top_k: Option<i32>,

        /// Top-p nucleus sampling
        #[arg(long)]
        top_p: Option<f32>,

        /// Model template (auto-detected from architecture if not specified)
        #[arg(long, value_enum)]
        template: Option<OllamaTemplate>,

        /// License text for the model
        #[arg(long)]
        license: Option<String>,
    },

    /// Create and register a model with Ollama
    Create {
        /// Model name for Ollama (e.g., "my-finetuned-model")
        #[arg(short, long)]
        name: String,

        /// Base model (GGUF path or Ollama model name)
        #[arg(short, long)]
        base: String,

        /// LoRA adapter path (optional)
        #[arg(long)]
        lora: Option<String>,

        /// System prompt
        #[arg(long)]
        system: Option<String>,

        /// Temperature (0.0-2.0)
        #[arg(long)]
        temperature: Option<f32>,

        /// Context window size
        #[arg(long)]
        num_ctx: Option<i32>,

        /// Model template (auto-detected from architecture if not specified)
        #[arg(long, value_enum)]
        template: Option<OllamaTemplate>,
    },

    /// List available templates
    Templates,
}

/// Ollama template presets.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum OllamaTemplate {
    /// Llama 3 chat format
    Llama3,
    /// Qwen3/ChatML format
    Qwen3,
    /// Gemma instruct format
    Gemma,
    /// Mistral instruct format
    Mistral,
    /// Phi-3 instruct format
    Phi3,
    /// DeepSeek chat format
    DeepSeek,
}

/// Discover `mlx.metallib` and set `PMETAL_METALLIB_PATH` so the patched MLX
/// C++ backend can find it regardless of where the binary is installed.
///
/// Search order: colocated → cache → Homebrew → auto-download → error.
#[allow(unsafe_code)]
fn ensure_metallib() {
    let metallib_name = "mlx.metallib";
    let cache_dir = std::env::var("HOME")
        .ok()
        .map(|h| PathBuf::from(h).join(".cache/pmetal/lib"));
    let mut search_paths: Vec<PathBuf> = Vec::new();

    // 1. Colocated with binary
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            search_paths.push(dir.join(metallib_name));
        }
    }

    // 1b. Search build directories (for local development)
    if let Ok(cwd) = std::env::current_dir() {
        // Search in common build output locations relative to workspace root
        let build_patterns = [
            "target/release/build",
            "target/debug/build",
            "target/aarch64-apple-darwin/release/build",
            "target/aarch64-apple-darwin/debug/build",
        ];

        for pattern in build_patterns {
            let build_dir = cwd.join(pattern);
            if build_dir.exists() {
                // Look for mlx-sys build output specifically
                if let Ok(entries) = std::fs::read_dir(build_dir) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if path.is_dir()
                            && path
                                .file_name()
                                .and_then(|n| n.to_str())
                                .is_some_and(|s| s.contains("pmetal-mlx-sys"))
                        {
                            let metallib_candidate = path.join("out/build/lib").join(metallib_name);
                            if metallib_candidate.is_file() {
                                search_paths.push(metallib_candidate);
                            }
                        }
                    }
                }
            }
        }
    }

    // 2. User cache ($HOME/.cache/pmetal/lib/) — matches C++ fallback path
    if let Some(ref cache) = cache_dir {
        search_paths.push(cache.join(metallib_name));
    }
    // 3. Homebrew Apple Silicon
    search_paths.push("/opt/homebrew/lib/mlx.metallib".into());
    // 4. Homebrew Intel / standard
    search_paths.push("/usr/local/lib/mlx.metallib".into());

    if let Some(path) = search_paths.iter().find(|p| p.is_file()) {
        // SAFETY: called before any other threads; env var is process-internal
        unsafe {
            std::env::set_var("PMETAL_METALLIB_PATH", path);
            // Q1 2026 Best Practice: Enable JIT for architecture-specific optimizations (M4+)
            std::env::set_var("MLX_METAL_JIT", "1");
        };
        tracing::debug!(path = %path.display(), "Found mlx.metallib");

        // Auto-cache: if found colocated with binary, copy to cache dir for resilience
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                if path.as_path() == dir.join(metallib_name) {
                    if let Some(ref cache) = cache_dir {
                        let dest = cache.join(metallib_name);
                        if !dest.exists() {
                            let _ = std::fs::create_dir_all(cache);
                            if std::fs::copy(path, &dest).is_ok() {
                                tracing::debug!("Cached metallib to {}", dest.display());
                            }
                        }
                    }
                }
            }
        }
        return;
    }

    // Not found locally — try to auto-download from GitHub releases
    if let Some(ref cache) = cache_dir {
        let dest = cache.join(metallib_name);
        if download_metallib(&dest) {
            unsafe { std::env::set_var("PMETAL_METALLIB_PATH", &dest) };
            return;
        }
    }

    // Download failed or no HOME — print actionable instructions
    let locations = search_paths
        .iter()
        .map(|p| format!("  - {}", p.display()))
        .collect::<Vec<_>>()
        .join("\n");
    eprintln!(
        "\n\x1b[1;33mwarning:\x1b[0m mlx.metallib not found — Metal GPU acceleration will fail.\n\n\
         Searched:\n{locations}\n\n\
         To fix this, do ONE of the following:\n\
         \x1b[1m  1. Rebuild from source:\x1b[0m  cargo install pmetal-cli  (auto-caches metallib)\n\
         \x1b[1m  2. Download manually:\x1b[0m\n\
             curl -fSL -o ~/.cache/pmetal/lib/mlx.metallib \\\n\
               https://github.com/epistates/pmetal/releases/download/v{version}/mlx.metallib\n\
         \x1b[1m  3. Homebrew:\x1b[0m             brew install epistates/tap/pmetal\n\n\
         The metallib ships with every pmetal release and is built during compilation.\n\
         See: https://github.com/epistates/pmetal#installation\n",
        version = env!("CARGO_PKG_VERSION"),
    );
}

/// Download `mlx.metallib` from the matching GitHub release.
///
/// Returns `true` if the download succeeded and the file is in place.
fn download_metallib(dest: &std::path::Path) -> bool {
    let version = env!("CARGO_PKG_VERSION");
    let url =
        format!("https://github.com/epistates/pmetal/releases/download/v{version}/mlx.metallib");

    eprintln!(
        "\x1b[1;36minfo:\x1b[0m mlx.metallib not found locally. \
         Downloading from GitHub releases (one-time setup)..."
    );

    if let Some(parent) = dest.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    // Use a temp file so a partial download doesn't leave a broken metallib
    let tmp = dest.with_extension("metallib.tmp");

    let status = std::process::Command::new("curl")
        .args([
            "-fSL",           // fail on HTTP errors, show errors, follow redirects
            "--progress-bar", // compact progress indicator
            "-o",
        ])
        .arg(&tmp)
        .arg(&url)
        .status();

    match status {
        Ok(s) if s.success() => {
            // Atomic-ish rename into place
            if std::fs::rename(&tmp, dest).is_ok() {
                eprintln!(
                    "\x1b[1;32minfo:\x1b[0m Cached mlx.metallib to {}\n",
                    dest.display()
                );
                true
            } else {
                let _ = std::fs::remove_file(&tmp);
                false
            }
        }
        _ => {
            let _ = std::fs::remove_file(&tmp);
            eprintln!(
                "\x1b[1;33mwarning:\x1b[0m Download failed (offline or release v{version} \
                 not published yet).\n"
            );
            false
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    ensure_metallib();

    let cli = Cli::parse();

    match cli.command {
        Commands::Train {
            config,
            model,
            dataset,
            eval_dataset,
            output,
            lora_r,
            lora_alpha,
            learning_rate,
            batch_size,
            epochs,
            max_seq_len,
            gradient_accumulation_steps,
            no_flash_attention,
            max_grad_norm,
            resume,
            quantization,
            quant_block_size,
            double_quant,
            no_fused,
            no_metal_fused_optimizer,
            no_sequence_packing,
            no_jit_compilation,
            no_gradient_checkpointing,
            gradient_checkpointing_layers,
            log_metrics,
            embedding_lr,
            #[cfg(feature = "ane")]
            no_ane,
        } => {
            // Optimizations enabled by default (invert no_* flags)
            let use_metal_flash_attention = !no_flash_attention;
            let fused = !no_fused;
            let use_metal_fused_optimizer = !no_metal_fused_optimizer;
            let use_sequence_packing = !no_sequence_packing;
            let use_jit_compilation = !no_jit_compilation;
            let gradient_checkpointing = !no_gradient_checkpointing;

            run_training(
                config,
                model,
                dataset,
                eval_dataset,
                output,
                lora_r,
                lora_alpha,
                learning_rate,
                batch_size,
                epochs,
                max_seq_len,
                gradient_accumulation_steps,
                use_metal_flash_attention,
                max_grad_norm,
                resume,
                quantization,
                quant_block_size,
                double_quant,
                fused,
                use_metal_fused_optimizer,
                use_sequence_packing,
                use_jit_compilation,
                gradient_checkpointing,
                gradient_checkpointing_layers,
                log_metrics,
                embedding_lr,
                #[cfg(feature = "ane")]
                !no_ane,
                #[cfg(not(feature = "ane"))]
                false,
            )
            .await?;
        }

        Commands::Infer {
            model,
            lora,
            prompt,
            max_tokens,
            temperature,
            top_k,
            top_p,
            min_p,
            repetition_penalty,
            frequency_penalty,
            presence_penalty,
            seed,
            chat,
            system,
            no_thinking,
            metal_sampler,
            compiled,
            stream,
            minimal,
            show_thinking,
            fp8,
            #[cfg(feature = "ane")]
            no_ane,
            #[cfg(feature = "ane")]
            ane_max_seq_len,
        } => {
            run_inference(
                &model,
                lora.as_deref(),
                &prompt,
                max_tokens,
                temperature,
                top_k,
                top_p,
                min_p,
                repetition_penalty,
                frequency_penalty,
                presence_penalty,
                seed,
                chat,
                system.as_deref(),
                no_thinking,
                metal_sampler,
                compiled,
                stream,
                minimal,
                show_thinking,
                fp8,
                #[cfg(feature = "ane")]
                !no_ane,
                #[cfg(not(feature = "ane"))]
                false,
                #[cfg(feature = "ane")]
                ane_max_seq_len,
                #[cfg(not(feature = "ane"))]
                1024,
            )
            .await?;
        }

        Commands::Download { model, revision } => {
            tracing::info!(model = %model, "Downloading model");
            let path = pmetal_hub::download_model(&model, revision.as_deref(), None).await?;
            println!("Model downloaded to: {}", path.display());
        }

        Commands::Memory => {
            let stats = pmetal_mlx::memory::get_memory_stats();
            println!("Memory Statistics:");
            println!("  Total:     {:.2} GB", stats.total_gb());
            if stats.total_gb() > 0.0 {
                let used_pct = (stats.used_gb() / stats.total_gb()) * 100.0;
                let peak_pct = (stats.peak_gb() / stats.total_gb()) * 100.0;
                println!("  Used:      {:.2} GB ({:.0}%)", stats.used_gb(), used_pct);
                println!(
                    "  Available: {:.2} GB ({:.0}%)",
                    stats.available_gb(),
                    100.0 - used_pct
                );
                println!("  Peak:      {:.2} GB ({:.0}%)", stats.peak_gb(), peak_pct);
            } else {
                println!("  Used:      {:.2} GB", stats.used_gb());
                println!("  Available: {:.2} GB", stats.available_gb());
                println!("  Peak:      {:.2} GB", stats.peak_gb());
            }
        }

        Commands::Bench {
            model,
            batch_size,
            seq_len,
        } => {
            run_benchmark(&model, batch_size, seq_len).await?;
        }

        Commands::Init { output } => {
            // Validate output path to prevent path traversal attacks
            let validated_output = validate_output_path(&output, "config output")?;
            generate_sample_config(&validated_output.to_string_lossy())?;
        }

        Commands::BenchFfi => {
            run_ffi_benchmark()?;
        }

        Commands::BenchGen { model } => {
            run_gen_benchmark(&model).await?;
        }

        Commands::Ollama { action } => {
            run_ollama_command(action).await?;
        }

        Commands::Fuse {
            model,
            lora,
            output,
            alpha,
            rank,
        } => {
            run_fuse(&model, &lora, &output, alpha, rank).await?;
        }

        Commands::Quantize {
            model,
            output,
            imatrix,
            method,
            lora,
        } => {
            if let Some(lora_path) = &lora {
                // Fuse LoRA first, then quantize the fused model
                let fused_dir = format!("{output}.fused_tmp");
                run_fuse(&model, lora_path, &fused_dir, None, None).await?;
                run_quantization(&fused_dir, &output, imatrix.as_deref(), &method).await?;
                // Clean up temp dir
                let _ = std::fs::remove_dir_all(&fused_dir);
            } else {
                run_quantization(&model, &output, imatrix.as_deref(), &method).await?;
            }
        }

        Commands::Distill {
            teacher,
            student,
            dataset,
            output,
            method,
            loss_type,
            temperature,
            alpha,
            rationale,
            rationale_weight,
            lora_r,
            learning_rate,
            batch_size,
            epochs,
            max_seq_len,
        } => {
            run_distillation_cli(
                &teacher,
                &student,
                &dataset,
                &output,
                &method,
                &loss_type,
                temperature,
                alpha,
                rationale,
                rationale_weight,
                lora_r,
                learning_rate,
                batch_size,
                epochs,
                max_seq_len,
            )
            .await?;
        }

        Commands::Grpo {
            model,
            dataset,
            output,
            num_generations,
            beta,
            learning_rate,
            max_seq_len,
            dapo,
            reasoning_rewards,
            no_flash_attention,
        } => {
            run_grpo_cli(
                &model,
                &dataset,
                &output,
                num_generations,
                beta,
                learning_rate,
                max_seq_len,
                dapo,
                reasoning_rewards,
                !no_flash_attention,
            )
            .await?;
        }

        Commands::Dataset { action } => {
            run_dataset_command(action).await?;
        }

        Commands::Dashboard { metrics_file } => {
            let path = metrics_file.map(std::path::PathBuf::from);
            dashboard::run_dashboard(path)?;
        }

        #[cfg(feature = "dashboard")]
        Commands::Tui { metrics_file } => {
            let path = metrics_file.map(std::path::PathBuf::from);
            tui::run(path).await?;
        }
    }

    Ok(())
}

/// Run model quantization.
async fn run_quantization(
    model_path: &str,
    output_path: &str,
    imatrix_path: Option<&str>,
    method: &str,
) -> anyhow::Result<()> {
    use pmetal_gguf::{
        GgmlType,
        GgufBuilder,
        dynamic::{DynamicQuantizationConfig, DynamicQuantizer},
        imatrix::IMatrix,
        quantize::quantize, // Import the function explicitly
    };
    use std::path::{Path, PathBuf};

    println!("========================================");
    println!("  PMetal GGUF Quantization");
    println!("========================================");
    println!("Model:    {}", model_path);
    println!("Output:   {}", output_path);
    println!("Method:   {}", method);
    if let Some(imp) = imatrix_path {
        println!("IMatrix:  {}", imp);
    }
    println!("========================================\n");

    // Resolve HuggingFace model ID to local path
    let resolved_model_path: PathBuf =
        if model_path.contains('/') && !PathBuf::from(model_path).exists() {
            // HuggingFace model ID - download/resolve to cache
            tracing::info!("Resolving HuggingFace model: {}", model_path);
            pmetal_hub::download_model(model_path, None, None).await?
        } else {
            PathBuf::from(model_path)
        };

    // 1. Validate quantization method (fail fast before any I/O)
    const VALID_METHODS: &[&str] = &["dynamic", "q8_0", "q4_k_m"];
    if !VALID_METHODS.contains(&method) {
        anyhow::bail!(
            "Unknown quantization method '{}'. Valid methods: {}",
            method,
            VALID_METHODS.join(", ")
        );
    }

    // 2. Load IMatrix if provided
    let imatrix = if let Some(path) = imatrix_path {
        tracing::info!("Loading IMatrix from {}", path);
        Some(IMatrix::load(Path::new(path))?)
    } else {
        None
    };

    // 3. Initialize Dynamic Quantizer
    let quantizer = if method == "dynamic" {
        let config = DynamicQuantizationConfig::default();
        DynamicQuantizer::new(config, imatrix)
    } else {
        let base_type = match method {
            "q8_0" => GgmlType::Q8_0,
            "q4_k_m" => GgmlType::Q4K,
            _ => unreachable!("validated above"),
        };

        let config = DynamicQuantizationConfig {
            base_type,
            high_precision_type: base_type,
            fallback_type: base_type,
            ..Default::default()
        };
        DynamicQuantizer::new(config, None)
    };

    // 4. Load Model Weights
    tracing::info!("Scanning model weights from {:?}...", resolved_model_path);
    // Use the loader from pmetal_models to handle sharded safetensors
    let weights = pmetal_models::loader::load_weights(&resolved_model_path)
        .map_err(|e| anyhow::anyhow!("Failed to load weights: {}", e))?;

    tracing::info!("Loaded {} tensors", weights.len());

    // 5. Detect Architecture
    let config_path = resolved_model_path.join("config.json");
    let mut architecture = "llama".to_string(); // Default fallback

    if config_path.exists() {
        if let Ok(content) = std::fs::read_to_string(&config_path) {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                if let Some(archs) = json.get("architectures").and_then(|v| v.as_array()) {
                    if let Some(arch_str) = archs.first().and_then(|v| v.as_str()) {
                        architecture = match arch_str {
                            "LlamaForCausalLM" => "llama".to_string(),
                            "MistralForCausalLM" => "mistral".to_string(),
                            "Qwen2ForCausalLM" => "qwen2".to_string(),
                            "GemmaForCausalLM" | "Gemma2ForCausalLM" => "gemma".to_string(),
                            "PhiForCausalLM" | "Phi3ForCausalLM" => "phi".to_string(),
                            // Add more mappings as needed
                            _ => {
                                tracing::warn!(
                                    "Unknown architecture '{}', defaulting to 'llama'",
                                    arch_str
                                );
                                "llama".to_string()
                            }
                        };
                        tracing::info!(
                            "Detected architecture: {} (from {})",
                            architecture,
                            arch_str
                        );
                    }
                }
            }
        }
    } else {
        tracing::warn!("config.json not found, defaulting architecture to 'llama'");
    }

    // 6. Initialize GGUF Builder
    let mut builder = GgufBuilder::with_model(&architecture, "quantized-model");

    // 7. Quantize and Write
    tracing::info!("Starting quantization...");

    // Sort keys for deterministic output
    let mut keys: Vec<_> = weights.keys().collect();
    keys.sort();

    for name in keys {
        let tensor = weights
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("Tensor {} not found in loaded weights", name))?;
        // Skip quantization for non-F32/F16 tensors (e.g. integer indices) if any
        // But most LLM weights are floats.

        let shape = tensor.shape();
        let shape_u64: Vec<u64> = shape.iter().map(|&d| d as u64).collect();

        // Determine target type
        let target_type = quantizer.get_tensor_type(name, &shape_u64);

        // Convert MLX array to host vector
        // Note: This requires evaluating the array and copying data to CPU
        // Ensure tensor is evaluated
        tensor
            .eval()
            .map_err(|e| anyhow::anyhow!("MLX eval error: {}", e))?;

        // We assume weights are float32 for quantization input.
        // If they are float16/bfloat16, we convert them.
        let data_f32: Vec<f32> = match tensor.dtype() {
            pmetal_mlx::Dtype::Float32 => tensor.as_slice::<f32>().to_vec(),
            pmetal_mlx::Dtype::Float16 | pmetal_mlx::Dtype::Bfloat16 => {
                let t_f32 = tensor
                    .as_dtype(pmetal_mlx::Dtype::Float32)
                    .map_err(|e| anyhow::anyhow!("Dtype conversion error: {}", e))?;
                t_f32
                    .eval()
                    .map_err(|e| anyhow::anyhow!("MLX eval error: {}", e))?;
                t_f32.as_slice::<f32>().to_vec()
            }
            _ => {
                tracing::warn!("Skipping non-float tensor: {}", name);
                continue;
            }
        };

        // Quantize
        tracing::info!("Quantizing {} to {:?}", name, target_type);
        let quantized_data = quantize(&data_f32, target_type)
            .map_err(|e| anyhow::anyhow!("Quantization error for {}: {:?}", name, e))?;

        // Add to GGUF
        builder.add_raw_tensor(name, shape_u64, target_type, quantized_data);
    }

    // Validate and write output file
    let validated_output = validate_output_path(output_path, "quantization output")?;
    let mut file = std::fs::File::create(&validated_output)?;
    builder.write(&mut file)?;

    println!("Quantization complete!");
    Ok(())
}

/// Fuse LoRA adapter weights into a base model and save as a complete model.
///
/// This performs weight-level fusion: loads base model weights and LoRA adapter
/// weights, computes `W_fused = W_base + (alpha/r) * (B @ A)` for each targeted
/// layer, copies all other files (config, tokenizer, etc.), and saves the result.
async fn run_fuse(
    model_path: &str,
    lora_path: &str,
    output_path: &str,
    alpha_override: Option<f32>,
    rank_override: Option<usize>,
) -> anyhow::Result<()> {
    use std::collections::HashMap;
    use std::path::Path;

    println!("  PMetal LoRA Fuse");
    println!("========================================");

    // Resolve model path (could be HF ID or local path)
    let model_dir: PathBuf = if model_path.contains('/') && !PathBuf::from(model_path).exists() {
        tracing::info!("Resolving HuggingFace model: {}", model_path);
        pmetal_hub::download_model(model_path, None, None).await?
    } else {
        PathBuf::from(model_path)
    };
    println!("Base model:   {}", model_dir.display());

    // Resolve LoRA adapter path
    let lora_file = if Path::new(lora_path).is_dir() {
        let f = Path::new(lora_path).join("lora_weights.safetensors");
        if !f.exists() {
            anyhow::bail!("No lora_weights.safetensors found in {}", lora_path);
        }
        f
    } else {
        PathBuf::from(lora_path)
    };
    println!("LoRA adapter: {}", lora_file.display());
    println!("Output:       {}", output_path);

    // Load base model weights
    print!("\nLoading base model weights... ");
    let mut base_weights = pmetal_models::loader::load_weights(&model_dir)?;
    println!("OK ({} tensors)", base_weights.len());

    // Load LoRA adapter weights
    print!("Loading LoRA adapter weights... ");
    let lora_weights =
        mlx_rs::Array::load_safetensors(&lora_file).map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("OK ({} tensors)", lora_weights.len());

    // Detect LoRA rank from adapter weights
    let rank = if let Some(r) = rank_override {
        r
    } else {
        // Find any lora_a weight to determine rank from its shape
        lora_weights
            .iter()
            .find(|(k, _)| k.contains("lora_a"))
            .map(|(_, v)| {
                let shape = v.shape();
                // lora_a is [r, in_features] or [in_features, r] depending on convention
                *shape.iter().min().unwrap_or(&16) as usize
            })
            .unwrap_or(16)
    };
    let alpha = alpha_override.unwrap_or(rank as f32);
    let scale = alpha / rank as f32;
    println!("LoRA rank: {rank}, alpha: {alpha}, scale: {scale:.3}");

    // Apply LoRA: W_fused = W_base + scale * (B @ A)
    print!("Fusing weights... ");
    let mut fused_count = 0usize;

    // Group LoRA weights by layer: find matching (lora_a, lora_b) pairs
    let mut lora_a_map: HashMap<String, &mlx_rs::Array> = HashMap::new();
    let mut lora_b_map: HashMap<String, &mlx_rs::Array> = HashMap::new();

    for (name, array) in &lora_weights {
        if let Some(base_name) = name.strip_suffix(".lora_a") {
            lora_a_map.insert(base_name.to_string(), array);
        } else if let Some(base_name) = name.strip_suffix(".lora_b") {
            lora_b_map.insert(base_name.to_string(), array);
        }
    }

    for (layer_name, lora_a) in &lora_a_map {
        let Some(lora_b) = lora_b_map.get(layer_name) else {
            tracing::warn!("Missing lora_b for {layer_name}, skipping");
            continue;
        };

        // Map LoRA layer name to base weight name
        // LoRA names: "layers.0.self_attn.q_proj.lora_a" → base: "model.layers.0.self_attn.q_proj.weight"
        let base_key = if layer_name.starts_with("model.") {
            format!("{layer_name}.weight")
        } else {
            format!("model.{layer_name}.weight")
        };

        let Some(base_weight) = base_weights.get(&base_key) else {
            tracing::warn!("Base weight {base_key} not found, skipping");
            continue;
        };

        // Compute delta = scale * (B @ A) and add to base weight
        // lora_a: [r, in_features], lora_b: [out_features, r]
        // delta: [out_features, in_features]
        let delta = mlx_rs::ops::matmul(lora_b, lora_a)?;
        let scaled_delta = mlx_rs::ops::multiply(&delta, mlx_rs::array!(scale))?;
        let fused = mlx_rs::ops::add(base_weight, &scaled_delta)?;

        base_weights.insert(base_key, fused);
        fused_count += 1;
    }
    println!("OK ({fused_count} layers fused)");

    // Create output directory
    let output_dir = Path::new(output_path);
    std::fs::create_dir_all(output_dir)?;

    // Copy non-weight files from base model (config.json, tokenizer, etc.)
    print!("Copying model files... ");
    let mut copied = 0usize;
    for entry in std::fs::read_dir(&model_dir)?.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        // Skip weight files (we'll save our own) and symlinks
        if name_str.ends_with(".safetensors")
            || name_str == "model.safetensors.index.json"
            || name_str.starts_with(".")
        {
            continue;
        }
        let dest = output_dir.join(&name);
        if entry.path().is_file() {
            std::fs::copy(entry.path(), &dest)?;
            copied += 1;
        }
    }
    println!("OK ({copied} files)");

    // Save fused weights as a single safetensors file
    print!("Saving fused model... ");
    let output_file = output_dir.join("model.safetensors");
    mlx_rs::Array::save_safetensors(&base_weights, None, &output_file)
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    let size = std::fs::metadata(&output_file)
        .map(|m| m.len())
        .unwrap_or(0);
    let gb = size as f64 / (1024.0 * 1024.0 * 1024.0);
    println!("OK ({:.2} GB)", gb);

    println!("\n========================================");
    println!("Fused model saved to: {output_path}");
    println!("\nNext steps:");
    println!(
        "  Inference:  pmetal infer -m {} -p \"Your prompt\"",
        output_path
    );
    println!(
        "  Quantize:   pmetal quantize -m {} -o {}.gguf",
        output_path, output_path
    );
    println!(
        "  Ollama:     pmetal ollama create -n my-model -b {}",
        output_path
    );

    Ok(())
}

/// Run knowledge distillation.
#[allow(clippy::too_many_arguments)]
async fn run_distillation_cli(
    teacher_id: &str,
    student_id: &str,
    dataset_path: &str,
    output_dir: &str,
    method_str: &str,
    loss_type_str: &str,
    temperature: f32,
    alpha: f32,
    rationale: bool,
    rationale_weight: f32,
    lora_r: usize,
    learning_rate: f32,
    batch_size: usize,
    epochs: usize,
    max_seq_len: usize,
) -> anyhow::Result<()> {
    use pmetal_core::LoraConfig;
    use pmetal_data::{DataLoaderConfig, DatasetFormat, Tokenizer, TrainingDataset};
    use pmetal_distill::{DistillConfig, DistillMethod, Distiller, LossConfig, LossType};
    use pmetal_lora::DynamicLoraModel;
    use pmetal_trainer::{DistillationTrainer, TrainingLoopConfig};
    use std::path::{Path, PathBuf};

    println!("========================================");
    println!("  PMetal Knowledge Distillation");
    println!("========================================");
    println!("Teacher:       {}", teacher_id);
    println!("Student:       {}", student_id);
    println!("Dataset:       {}", dataset_path);
    println!("Output:        {}", output_dir);
    println!("Method:        {}", method_str);
    println!("Loss Type:     {}", loss_type_str);
    println!("Temperature:   {}", temperature);
    println!("Alpha:         {}", alpha);
    if rationale {
        println!("Rationale:     enabled (weight: {})", rationale_weight);
    }
    println!("LoRA Rank:     {}", lora_r);
    println!("LR:            {:.2e}", learning_rate);
    println!("Batch Size:    {}", batch_size);
    println!("Epochs:        {}", epochs);
    println!("Max Seq Len:   {}", max_seq_len);
    println!("========================================\n");

    // 1. Resolve and Download Models
    tracing::info!("Resolving teacher model: {}", teacher_id);
    let teacher_path = if teacher_id.contains('/') && !Path::new(teacher_id).exists() {
        pmetal_hub::download_model(teacher_id, None, None).await?
    } else {
        PathBuf::from(teacher_id)
    };

    tracing::info!("Resolving student model: {}", student_id);
    let student_path = if student_id.contains('/') && !Path::new(student_id).exists() {
        pmetal_hub::download_model(student_id, None, None).await?
    } else {
        PathBuf::from(student_id)
    };

    // 2. Load Tokenizer (from student, with config-aware special token resolution)
    tracing::info!("Loading tokenizer...");
    let tokenizer = Tokenizer::from_model_dir(&student_path)?;

    // 2b. Detect chat template from student model
    let chat_template =
        pmetal_data::chat_templates::detect_chat_template(&student_path, student_id);

    // 3. Load Dataset
    tracing::info!("Loading dataset: {}", dataset_path);
    let train_dataset = TrainingDataset::from_jsonl_tokenized(
        dataset_path,
        &tokenizer,
        DatasetFormat::Auto,
        max_seq_len,
        Some(&chat_template),
    )?;

    // 4. Load Teacher Model (Frozen)
    tracing::info!("Loading teacher model...");
    let teacher_lora_config = LoraConfig {
        r: 0,
        ..Default::default()
    };
    let mut teacher_model = DynamicLoraModel::from_pretrained(&teacher_path, teacher_lora_config)?;

    // 5. Load Student Model (Trainable LoRA)
    tracing::info!("Loading student model...");
    let student_lora_config = LoraConfig {
        r: lora_r,
        alpha: (lora_r * 2) as f32,
        ..Default::default()
    };
    let mut student_model = DynamicLoraModel::from_pretrained(&student_path, student_lora_config)?;

    // 6. Setup Distillation Engine
    let method = match method_str.to_lowercase().as_str() {
        "online" => DistillMethod::Online,
        "offline" => DistillMethod::Offline,
        "progressive" => DistillMethod::Progressive,
        _ => DistillMethod::Online,
    };

    let loss_type = match loss_type_str.to_lowercase().as_str() {
        "kl_divergence" => LossType::KlDivergence,
        "jensen_shannon" => LossType::JensenShannon,
        "soft_cross_entropy" => LossType::SoftCrossEntropy,
        "mse_loss" => LossType::MseLoss,
        _ => LossType::KlDivergence,
    };

    let validated_distill_output = validate_output_path(output_dir, "distillation output")?;
    let distill_config = DistillConfig {
        teacher: teacher_id.to_string(),
        student: student_id.to_string(),
        method,
        loss: LossConfig {
            loss_type,
            temperature,
            alpha,
            rationale,
            rationale_weight,
            ..Default::default()
        },
        offline: None,
        output_path: Some(validated_distill_output.clone()),
        training: pmetal_distill::TrainingConfig {
            batch_size,
            learning_rate,
            epochs,
            max_seq_len,
            ..Default::default()
        },
    };

    let distiller = Distiller::new(distill_config)?;

    // 7. Setup Trainer
    let training_loop_config = TrainingLoopConfig {
        training: pmetal_core::TrainingConfig {
            learning_rate: learning_rate as f64,
            batch_size,
            num_epochs: epochs,
            max_seq_len,
            output_dir: validated_distill_output.display().to_string(),
            ..Default::default()
        },
        dataloader: DataLoaderConfig {
            batch_size,
            max_seq_len,
            shuffle: true,
            seed: 42,
            pad_token_id: tokenizer.pad_token_id().unwrap_or(0),
            drop_last: false,
        },
        use_metal_flash_attention: true,
        log_every: 1,
        checkpoint_every: 100,
        eval_every: 0,
        use_jit_compilation: true,
        use_sequence_packing: true,
        gradient_checkpointing: true,
        gradient_checkpointing_layers: 4,
        embedding_lr: None,
        eager_evaluation: false,
        use_metal_fused_optimizer: true,
    };

    let mut trainer = DistillationTrainer::new(distiller, training_loop_config);

    // 8. Run Distillation
    trainer
        .run(
            &mut student_model,
            &mut teacher_model,
            train_dataset,
            None,
            None,
        )
        .map_err(|e| anyhow::anyhow!("Distillation failed: {}", e))?;

    // 9. Save Student Adapters
    let lora_output = validated_distill_output.join("lora_weights.safetensors");
    tracing::info!("Saving distilled student adapters to {:?}", lora_output);
    std::fs::create_dir_all(&validated_distill_output)?;
    student_model.save_lora_weights(&lora_output)?;

    println!("\n========================================");
    println!("  Distillation Complete");
    println!("========================================");
    println!("  Adapters:  {}", lora_output.display());
    println!("  Student:   {}", student_id);
    println!("  Template:  {:?}", chat_template.template_type);
    println!("========================================");
    println!("\nNext steps:");
    println!(
        "  Inference:  pmetal run {} --lora {}",
        student_id,
        lora_output.display()
    );
    Ok(())
}

/// Run GRPO (Group Relative Policy Optimization) for reasoning models.
#[allow(clippy::too_many_arguments)]
async fn run_grpo_cli(
    model_id: &str,
    dataset_path: &str,
    output_dir: &str,
    num_generations: usize,
    beta: f64,
    learning_rate: f64,
    max_seq_len: usize,
    dapo: bool,
    reasoning_rewards: bool,
    use_metal_flash_attention: bool,
) -> anyhow::Result<()> {
    use std::path::{Path, PathBuf};

    println!("========================================");
    println!("  PMetal GRPO Reasoning Training");
    println!("========================================");
    println!("Model:         {}", model_id);
    println!("Dataset:       {}", dataset_path);
    println!("Output:        {}", output_dir);
    println!("Generations:   {}", num_generations);
    println!("Beta:          {}", beta);
    println!("LR:            {:.2e}", learning_rate);
    println!("DAPO:          {}", dapo);
    println!("Reasoning Rew: {}", reasoning_rewards);
    println!("========================================\n");

    // 1. Resolve and Download Model
    tracing::info!("Resolving model: {}", model_id);
    let model_path = if model_id.contains('/') && !Path::new(model_id).exists() {
        pmetal_hub::download_model(model_id, None, None).await?
    } else {
        PathBuf::from(model_id)
    };

    // 2. Load Tokenizer (with config-aware special token resolution)
    tracing::info!("Loading tokenizer...");
    let tokenizer = Tokenizer::from_model_dir(&model_path)?;

    // 2b. Detect chat template
    let chat_template = pmetal_data::chat_templates::detect_chat_template(&model_path, model_id);

    // 3. Load Dataset (Prompts)
    tracing::info!("Loading prompt dataset: {}", dataset_path);
    let dataset = TrainingDataset::from_jsonl_tokenized(
        dataset_path,
        &tokenizer,
        DatasetFormat::Auto,
        max_seq_len,
        Some(&chat_template),
    )?;

    // 4. Load Model (Trainable LoRA)
    tracing::info!("Loading model with LoRA...");
    let lora_config = LoraConfig {
        r: 16,
        alpha: 32.0,
        ..Default::default()
    };
    let mut model = DynamicLoraModel::from_pretrained(&model_path, lora_config)?;

    // 5. Setup GRPO Config
    let mut grpo_config = GrpoConfig::new(num_generations).with_beta(beta);

    if dapo {
        grpo_config = grpo_config.for_dapo();
    }

    // 6. Setup Reward Functions
    let mut rewards = pmetal_trainer::CombinedReward::new();

    if reasoning_rewards {
        // Simple reasoning format reward: favors completions with <thinking> tags
        struct FormatReward;
        impl pmetal_trainer::RewardFunction for FormatReward {
            fn compute(
                &self,
                _prompts: &[String],
                completions: &[String],
                _images: Option<&[Vec<mlx_rs::Array>]>,
            ) -> pmetal_trainer::GrpoResult<Vec<f64>> {
                Ok(completions
                    .iter()
                    .map(|c| {
                        if c.contains("<thinking>") && c.contains("</thinking>") {
                            1.0
                        } else if c.contains("<thinking>") {
                            0.5
                        } else {
                            0.0
                        }
                    })
                    .collect())
            }
            fn name(&self) -> &str {
                "format_reward"
            }
        }

        // Length reward: small penalty for being too long or too short
        struct LengthReward(usize);
        impl pmetal_trainer::RewardFunction for LengthReward {
            fn compute(
                &self,
                _prompts: &[String],
                completions: &[String],
                _images: Option<&[Vec<mlx_rs::Array>]>,
            ) -> pmetal_trainer::GrpoResult<Vec<f64>> {
                Ok(completions
                    .iter()
                    .map(|c| {
                        let len = c.len();
                        if len > self.0 {
                            -0.1
                        } else if len < 10 {
                            -0.5
                        } else {
                            0.1
                        }
                    })
                    .collect())
            }
            fn name(&self) -> &str {
                "length_reward"
            }
        }

        rewards = rewards
            .add(Box::new(FormatReward), 1.0)
            .add(Box::new(LengthReward(max_seq_len * 2)), 0.2);
    } else {
        // Default dummy reward if none specified
        struct DummyReward;
        impl pmetal_trainer::RewardFunction for DummyReward {
            fn compute(
                &self,
                _p: &[String],
                completions: &[String],
                _i: Option<&[Vec<mlx_rs::Array>]>,
            ) -> pmetal_trainer::GrpoResult<Vec<f64>> {
                Ok(vec![0.1; completions.len()])
            }
            fn name(&self) -> &str {
                "dummy"
            }
        }
        rewards = rewards.add(Box::new(DummyReward), 1.0);
    }

    // 7. Setup Trainer
    let training_config = TrainingConfig {
        learning_rate,
        batch_size: 1, // GRPO generates num_generations per prompt, so batch_size 1 is typical
        num_epochs: 1,
        max_seq_len,
        output_dir: output_dir.to_string(),
        ..Default::default()
    };

    let _training_loop_config = TrainingLoopConfig {
        training: training_config.clone(),
        dataloader: DataLoaderConfig {
            batch_size: 1,
            max_seq_len,
            shuffle: true,
            seed: 42,
            pad_token_id: tokenizer.pad_token_id().unwrap_or(0),
            drop_last: false,
        },
        use_metal_flash_attention,
        log_every: 1,
        checkpoint_every: 50,
        eval_every: 0,
        use_jit_compilation: true,
        use_sequence_packing: false, // GRPO usually doesn't pack generations
        gradient_checkpointing: true,
        gradient_checkpointing_layers: 4,
        embedding_lr: None,
        eager_evaluation: true, // GRPO generates first, then trains - eager helps memory
        use_metal_fused_optimizer: true,
    };

    let mut trainer = GrpoTrainer::new(grpo_config, training_config)?;

    // 7b. Setup Optimizer
    let mut optimizer = mlx_rs::optimizers::AdamWBuilder::new(learning_rate as f32)
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to build optimizer: {}", e))?;

    // 8. Run Training
    println!("Starting GRPO training loop...");

    // Load reference model (frozen)
    println!("Loading reference model...");
    let mut ref_model = DynamicModel::load(&model_path)
        .map_err(|e| anyhow::anyhow!("Failed to load reference model: {}", e))?;

    trainer
        .run(
            &mut model,
            Some(&mut ref_model),
            &tokenizer,
            &dataset,
            &rewards,
            &mut optimizer,
        )
        .map_err(|e| anyhow::anyhow!("GRPO training error: {}", e))?;

    println!(
        "\nGRPO training complete! Model weights saved to: {}",
        output_dir
    );
    Ok(())
}

/// Run LoRA/QLoRA fine-tuning using the new TrainingLoop.
#[allow(clippy::too_many_arguments)]
async fn run_training(
    config_path: Option<String>,
    model_id: Option<String>,
    dataset_path: Option<String>,
    eval_dataset_path: Option<String>,
    output_dir: String,
    lora_r: usize,
    lora_alpha: f32,
    learning_rate: f64,
    batch_size: usize,
    num_epochs: usize,
    max_seq_len: usize,
    gradient_accumulation_steps: usize,
    use_metal_flash_attention: bool,
    max_grad_norm: f64,
    resume: bool,
    quantization: QuantizationMethod,
    quant_block_size: usize,
    double_quant: bool,
    fused: bool,
    use_metal_fused_optimizer: bool,
    use_sequence_packing: bool,
    use_jit_compilation: bool,
    gradient_checkpointing: bool,
    gradient_checkpointing_layers: usize,
    log_metrics: Option<String>,
    embedding_lr: Option<f32>,
    ane: bool,
) -> anyhow::Result<()> {
    #[cfg(not(feature = "ane"))]
    if ane {
        anyhow::bail!("ANE training requires the 'ane' feature: cargo build --features ane");
    }
    #[cfg(feature = "ane")]
    if ane {
        use pmetal_trainer::{AneTrainingLoop, AneTrainingLoopConfig, DynamicAneTrainerConfig};

        tracing::info!("Attempting ANE dynamic weight pipeline");

        let ane_result: anyhow::Result<()> = async {
            // Resolve model path
            let model_name = model_id
                .as_deref()
                .or(config_path.as_deref())
                .ok_or_else(|| anyhow::anyhow!("--model is required for ANE training"))?;
            let model_path = if model_name.contains('/') && !PathBuf::from(model_name).exists() {
                pmetal_hub::download_model(model_name, None, None).await?
            } else {
                PathBuf::from(model_name)
            };

            // Read model config
            let config_text = std::fs::read_to_string(model_path.join("config.json"))?;
            let config_json: serde_json::Value = serde_json::from_str(&config_text)?;

            // Validate architecture compatibility before attempting ANE
            if let Err(reason) = DynamicAneTrainerConfig::is_ane_compatible(&config_json) {
                anyhow::bail!("{}", reason);
            }

            let get_usize = |key: &str| -> anyhow::Result<usize> {
                config_json
                    .get(key)
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize)
                    .ok_or_else(|| anyhow::anyhow!("config.json missing '{key}'"))
            };
            let dim = get_usize("hidden_size")?;
            let hidden_dim = get_usize("intermediate_size")?;
            let n_heads = get_usize("num_attention_heads")?;
            let n_kv_heads = config_json
                .get("num_key_value_heads")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(n_heads);
            let n_layers = get_usize("num_hidden_layers")?;
            let vocab_size = get_usize("vocab_size")?;

            // Auto-detect max_seq_len from model config if not specified (0)
            let max_seq_len = if max_seq_len == 0 {
                let model_max = config_json
                    .get("max_position_embeddings")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize)
                    .unwrap_or(2048);
                // ANE kernels compile seq_len into static shapes — cap to prevent
                // excessively large IOSurface allocations.
                let capped = model_max.min(2048);
                tracing::info!(
                    model_max = model_max,
                    capped = capped,
                    "Auto-detected ANE seq_len from max_position_embeddings"
                );
                capped
            } else {
                max_seq_len
            };

            // Load tokenizer (with config-aware special token resolution) and dataset
            let tokenizer = Tokenizer::from_model_dir(&model_path)?;
            let ds_path = dataset_path
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("--dataset is required for ANE training"))?;
            let text_samples =
                TrainingDataset::load_jsonl_text(ds_path, DatasetFormat::Auto, None)?;

            // Tokenize and prepare batches: Vec<Vec<(Vec<u16>, Vec<u16>)>>
            // Each batch contains `gradient_accumulation_steps` examples.
            // Each example is (input, target) where target = input shifted by 1.
            let mut batches: Vec<Vec<(Vec<u16>, Vec<u16>)>> = Vec::new();
            let mut current_batch: Vec<(Vec<u16>, Vec<u16>)> = Vec::new();

            for sample in &text_samples {
                let tokens = tokenizer.encode(&sample.text)?;
                if tokens.len() < 2 {
                    continue;
                }
                let len = tokens.len().min(max_seq_len + 1);
                let input: Vec<u16> = tokens[..len - 1].iter().map(|&t| t as u16).collect();
                let target: Vec<u16> = tokens[1..len].iter().map(|&t| t as u16).collect();
                current_batch.push((input, target));

                if current_batch.len() >= gradient_accumulation_steps.max(1) {
                    batches.push(std::mem::take(&mut current_batch));
                }
            }
            if !current_batch.is_empty() {
                batches.push(current_batch);
            }

            if batches.is_empty() {
                anyhow::bail!("No training examples after tokenization");
            }

            let total_batches = batches.len() * num_epochs;
            tracing::info!(
                examples = text_samples.len(),
                batches = batches.len(),
                epochs = num_epochs,
                total_steps = total_batches,
                "Prepared ANE training data"
            );

            // Read head_dim from config (models like Qwen3 use non-standard head_dim)
            let head_dim = config_json
                .get("head_dim")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize);

            let trainer_config = DynamicAneTrainerConfig {
                dim,
                hidden_dim,
                n_heads,
                n_kv_heads,
                head_dim,
                n_layers,
                vocab_size,
                seq_len: max_seq_len,
                learning_rate: learning_rate as f32,
                accum_steps: gradient_accumulation_steps.max(1),
                ..Default::default()
            };

            let loop_config = AneTrainingLoopConfig {
                trainer: trainer_config,
                num_batches: total_batches,
                max_steps: total_batches,
                log_every: 10,
                save_every: Some(100),
                output_dir: PathBuf::from(&output_dir),
            };

            let mut training_loop = AneTrainingLoop::new(loop_config);

            // Load model weights from SafeTensors
            tracing::info!("Loading model weights for ANE training...");
            training_loop.load_weights_safetensors(&model_path)?;

            // Install vocab compaction (scan all batch tokens, build compact map)
            {
                use pmetal_trainer::VocabMap;
                let vocab_map = VocabMap::from_batches(&batches, vocab_size);
                tracing::info!(
                    compact_vocab = vocab_map.compact_vocab,
                    full_vocab = vocab_size,
                    "Vocab compaction ready"
                );
                training_loop.install_vocab_map(vocab_map);
            }

            // Compile dynamic ANE kernels (one-time, no recompilation ever)
            tracing::info!("Compiling dynamic ANE kernels (one-time)...");
            training_loop.compile_kernels()?;

            // Run training
            for epoch in 0..num_epochs {
                tracing::info!(epoch = epoch + 1, total = num_epochs, "Starting epoch");
                let state = training_loop.train(&batches)?;
                tracing::info!(
                    loss = state.loss,
                    tokens = state.tokens_processed,
                    tok_per_sec = format!("{:.1}", state.tokens_per_sec()),
                    "Epoch complete"
                );
            }

            tracing::info!("ANE training complete");
            Ok(())
        }
        .await;

        match ane_result {
            Ok(()) => return Ok(()),
            Err(e) => {
                tracing::warn!("ANE training failed ({}), falling back to GPU training", e);
            }
        }
    }

    let use_qlora = !matches!(quantization, QuantizationMethod::None);

    // Validate output directory to prevent path traversal attacks
    let validated_output = validate_output_path(&output_dir, "output directory")?;
    let output_dir = validated_output.to_string_lossy().to_string();

    // Load or create configuration
    let mut config = if let Some(ref path) = config_path {
        let content = std::fs::read_to_string(path)?;
        serde_yaml::from_str(&content)?
    } else {
        FullTrainingConfig::default()
    };

    // Override with CLI args if provided
    if let Some(ref model) = model_id {
        config.model.model_id = model.clone();
    }
    if let Some(ref ds) = dataset_path {
        config.dataset.dataset_id = ds.clone();
    }
    config.lora.r = lora_r;
    config.lora.alpha = lora_alpha;
    config.training.learning_rate = learning_rate;
    config.training.batch_size = batch_size;
    config.training.num_epochs = num_epochs;
    config.training.max_seq_len = max_seq_len;
    config.training.gradient_accumulation_steps = gradient_accumulation_steps;
    config.training.max_grad_norm = max_grad_norm;
    config.training.output_dir = output_dir.clone();

    // Download model if needed
    tracing::info!("Loading model: {}", config.model.model_id);
    let model_path =
        if config.model.model_id.contains('/') && !PathBuf::from(&config.model.model_id).exists() {
            // HuggingFace model ID
            pmetal_hub::download_model(
                &config.model.model_id,
                config.model.revision.as_deref(),
                None,
            )
            .await?
        } else {
            PathBuf::from(&config.model.model_id)
        };

    // Load model config (optional for GGUF - config is extracted from metadata)
    let model_config_path = model_path.join("config.json");
    let llama_config: Option<LlamaConfig> = if model_config_path.exists() {
        let content = std::fs::read_to_string(&model_config_path)?;
        Some(serde_json::from_str(&content)?)
    } else {
        // GGUF files don't have separate config.json
        if WeightFormat::detect(&model_path) != Some(WeightFormat::Gguf) {
            anyhow::bail!(
                "Model config.json not found at {:?}. If using GGUF, pass the .gguf file directly.",
                model_config_path
            );
        }
        None
    };

    // Auto-detect max_seq_len if requested (0)
    if config.training.max_seq_len == 0 {
        if let Some(ref cfg) = llama_config {
            // Llama/Qwen config uses max_position_embeddings — cap to prevent OOM
            let model_max = cfg.max_position_embeddings as usize;
            config.training.max_seq_len = model_max.min(8192);
            tracing::info!(
                "Auto-detected max_seq_len: {} (model supports {}, capped at 8192)",
                config.training.max_seq_len,
                model_max
            );
        } else {
            // GGUF fallback
            config.training.max_seq_len = 8192;
            tracing::info!(
                "Defaulting max_seq_len to {} (GGUF or unknown config)",
                config.training.max_seq_len
            );
        }
    }

    // Initialize metrics callback if requested
    let metrics_path_resolved = log_metrics.as_ref().map(|metrics_path| {
        if metrics_path.contains('/') || metrics_path.contains('\\') {
            PathBuf::from(metrics_path)
        } else {
            PathBuf::from(&output_dir).join(metrics_path)
        }
    });
    let mut metrics_callback: Option<Box<dyn pmetal_core::TrainingCallback>> =
        if let Some(ref path) = metrics_path_resolved {
            let callback = MetricsJsonCallback::new(path)?
                .with_run_name(format!(
                    "{}-{}",
                    config.model.model_id.replace('/', "-"),
                    chrono::Utc::now().format("%Y%m%d-%H%M%S")
                ))
                .with_config(serde_json::json!({
                    "model": config.model.model_id,
                    "lora_r": lora_r,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "epochs": num_epochs,
                    "max_seq_len": config.training.max_seq_len,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "gradient_checkpointing": gradient_checkpointing,
                    "quantization": format!("{:?}", quantization),
                }));
            use pmetal_core::TrainingCallback;
            let mut cb = callback;
            cb.on_train_start();
            Some(Box::new(cb) as Box<dyn pmetal_core::TrainingCallback>)
        } else {
            None
        };

    if let Some(ref cfg) = llama_config {
        tracing::info!(
            "Model: {} hidden, {} layers, {} heads",
            cfg.hidden_size,
            cfg.num_hidden_layers,
            cfg.num_attention_heads
        );
    } else {
        tracing::info!("Model config will be extracted from GGUF metadata");
    }

    // Load tokenizer (with config-aware special token resolution)
    tracing::info!("Loading tokenizer...");
    let tokenizer_path = model_path.join("tokenizer.json");
    let tokenizer = if tokenizer_path.exists() {
        Tokenizer::from_model_dir(&model_path)?
    } else {
        anyhow::bail!(
            "Tokenizer not found at {:?}. GGUF models don't bundle a tokenizer — \
             download the source model first with: pmetal download {}",
            tokenizer_path,
            config.model.model_id
        );
    };

    // Detect chat template for OpenAI/ShareGPT formatting (checks tokenizer_config.json first)
    let chat_template =
        pmetal_data::chat_templates::detect_chat_template(&model_path, &config.model.model_id);

    // Resolve dataset source — local path or HuggingFace dataset ID
    let dataset_path_resolved = match resolve_dataset_source(&config.dataset.dataset_id) {
        DatasetSource::Local(p) => p,
        DatasetSource::HuggingFace(id) => {
            tracing::info!("Downloading dataset from HuggingFace: {}", id);
            // Download the dataset repo (contains JSONL/Parquet files)
            let dir = pmetal_hub::download_dataset(&id, None, None, None).await?;
            // Try to resolve a JSONL file in the downloaded directory
            let resolved = TrainingDataset::resolve_dataset_path_pub(&dir);
            if let Ok(p) = resolved {
                p
            } else {
                // Fall back to parquet
                let parquet_paths =
                    pmetal_hub::download_dataset_parquet(&id, "train", None, None).await?;
                if parquet_paths.is_empty() {
                    anyhow::bail!("No JSONL or Parquet files found for dataset {}", id);
                }
                parquet_paths[0].clone()
            }
        }
    };

    // Load and tokenize training dataset — dispatch on file extension
    tracing::info!(
        "Loading training dataset: {}",
        dataset_path_resolved.display()
    );
    let is_parquet = dataset_path_resolved
        .extension()
        .is_some_and(|ext| ext == "parquet");
    let train_dataset = if is_parquet {
        tracing::info!("Detected Parquet format");
        // Try "text" column first, then fall back to common alternatives
        let result = TrainingDataset::from_parquet_tokenized(
            &dataset_path_resolved,
            &tokenizer,
            "text",
            config.training.max_seq_len,
            None,
        );
        match result {
            Ok(ds) => ds,
            Err(_) => {
                // Try "content" column
                TrainingDataset::from_parquet_tokenized(
                    &dataset_path_resolved,
                    &tokenizer,
                    "content",
                    config.training.max_seq_len,
                    None,
                )
                .map_err(|_| {
                    anyhow::anyhow!(
                        "Parquet file does not have a recognized column layout. \
                     Supported: 'text' column, 'content' column, or reasoning format \
                     (problem/thinking/solution columns)."
                    )
                })?
            }
        }
    } else {
        TrainingDataset::from_jsonl_tokenized(
            &dataset_path_resolved,
            &tokenizer,
            DatasetFormat::Auto,
            config.training.max_seq_len,
            Some(&chat_template),
        )?
    };
    tracing::info!("Training dataset loaded: {} samples", train_dataset.len());

    // Load evaluation dataset if provided
    let eval_dataset = if let Some(ref eval_path) = eval_dataset_path {
        tracing::info!("Loading evaluation dataset: {}", eval_path);
        let ds = TrainingDataset::from_jsonl_tokenized(
            eval_path,
            &tokenizer,
            DatasetFormat::Auto,
            config.training.max_seq_len,
            Some(&chat_template),
        )?;
        tracing::info!("Evaluation dataset loaded: {} samples", ds.len());
        Some(ds)
    } else {
        None
    };

    // Set up checkpointing
    let checkpoint_dir = PathBuf::from(&output_dir).join("checkpoints");
    let checkpoint_manager = CheckpointManager::new(&checkpoint_dir)?.with_max_checkpoints(3);

    // Create data loader config
    let dataloader_config = DataLoaderConfig {
        batch_size: config.training.batch_size,
        max_seq_len: config.training.max_seq_len,
        shuffle: config.dataset.shuffle,
        seed: config.training.seed,
        pad_token_id: tokenizer.pad_token_id().unwrap_or(0),
        drop_last: false,
    };

    // Create training loop config
    let training_loop_config = TrainingLoopConfig {
        training: config.training.clone(),
        dataloader: dataloader_config.clone(),
        use_metal_flash_attention,
        log_every: config.training.logging_steps,
        checkpoint_every: config.training.save_steps.unwrap_or(500),
        eval_every: if eval_dataset.is_some() { 100 } else { 0 },
        use_jit_compilation,
        use_sequence_packing,
        gradient_checkpointing,
        gradient_checkpointing_layers,
        embedding_lr,
        // Eager evaluation: forces immediate GPU computation after each step.
        // Prevents Metal resource exhaustion from deferred evaluation graph buildup.
        // Essential for models without true gradient checkpointing (e.g., Qwen3.5 hybrid).
        eager_evaluation: true,
        use_metal_fused_optimizer,
    };

    // Calculate total steps for progress bar
    let steps_per_epoch = train_dataset.len() / config.training.batch_size;
    let total_steps = if let Some(max) = config.training.max_steps {
        max
    } else {
        steps_per_epoch * config.training.num_epochs
    };

    // Set up progress bar
    let progress = ProgressBar::new(total_steps as u64);
    progress.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) | Loss: {msg}")
            .expect("Stop tokens should format correctly")
            .progress_chars("#>-"),
    );

    // Run training with either LoRA or QLoRA model
    let (final_loss, final_step, total_tokens) = if use_qlora {
        // QLoRA path - quantized base weights
        let quant_scheme = match quantization {
            QuantizationMethod::Nf4 => QuantScheme::NF4,
            QuantizationMethod::Fp4 => QuantScheme::FP4,
            QuantizationMethod::Int8 => QuantScheme::Int8,
            QuantizationMethod::None => unreachable!(),
        };

        let qlora_config = QLoraConfig {
            lora: config.lora.clone(),
            quant_scheme,
            block_size: quant_block_size,
            double_quant,
            compute_in_half: true,
        };

        tracing::info!(
            "Initializing QLoRA model with {:?} quantization...",
            quantization
        );
        // QLoRA currently requires config.json (Llama-only)
        let llama_cfg = llama_config.ok_or_else(|| {
            anyhow::anyhow!(
                "QLoRA requires config.json. GGUF format is only supported with standard LoRA."
            )
        })?;
        let mut model = LlamaQloraForCausalLM::with_qlora_config(llama_cfg, qlora_config)?;

        // Load and quantize base model weights
        tracing::info!(
            "Loading and quantizing base model weights from {:?}...",
            model_path
        );
        model.load_and_quantize_from_dir(&model_path)?;

        // Report memory savings
        let savings = model.memory_savings();
        let (quant_bytes, lora_bytes, total_bytes) = model.memory_usage();
        tracing::info!(
            "Memory usage: {:.2} MB (quantized: {:.2} MB, LoRA: {:.2} MB) - {:.1}% of full precision",
            total_bytes as f64 / 1_000_000.0,
            quant_bytes as f64 / 1_000_000.0,
            lora_bytes as f64 / 1_000_000.0,
            savings * 100.0
        );

        tracing::info!(
            "Trainable parameters: {}",
            format_param_count(model.num_trainable_params())
        );

        // Enable gradient checkpointing if requested
        if gradient_checkpointing {
            use pmetal_lora::TrainableModel;
            if model.supports_gradient_checkpointing() {
                model.enable_gradient_checkpointing(gradient_checkpointing_layers);
                tracing::info!(
                    "Gradient checkpointing enabled ({} layers per block)",
                    gradient_checkpointing_layers
                );
            } else {
                tracing::warn!(
                    "Gradient checkpointing requested but not supported by LlamaQloraForCausalLM. \
                     This feature is currently only supported for Qwen3 models."
                );
            }
        }

        // Create training loop
        let mut training_loop = TrainingLoop::new(training_loop_config);

        // Wire metrics callback into training loop for step-level dispatch
        if let Some(cb) = metrics_callback.take() {
            training_loop.add_callback(cb);
        }

        // Resume from checkpoint if requested
        if resume {
            if let Some((lora_params, metadata)) = checkpoint_manager.load_latest()? {
                tracing::info!("Resuming from checkpoint at step {}", metadata.step);
                model.set_lora_parameters(&lora_params);
                training_loop.set_step(metadata.step);
                training_loop.set_epoch(metadata.epoch);
            } else {
                tracing::info!("No checkpoint found, starting fresh");
            }
        }

        tracing::info!("Starting QLoRA training...");

        if fused {
            tracing::warn!(
                "Fused training is not yet supported for QLoRA, using standard training"
            );
        }

        // Run training loop
        training_loop.run(
            &mut model,
            train_dataset,
            eval_dataset,
            Some(&checkpoint_manager),
        )?;

        progress.finish_with_message(format!("{:.4}", training_loop.current_loss()));

        // Save final LoRA weights
        let final_path = PathBuf::from(&output_dir).join("lora_weights.safetensors");
        model.save_lora_weights(&final_path)?;
        tracing::info!("Saved LoRA weights to {:?}", final_path);

        // Recover metrics callback from training loop for finalization
        let mut cbs = training_loop.take_callbacks();
        if !cbs.is_empty() && metrics_callback.is_none() {
            metrics_callback = Some(cbs.remove(0));
        }

        (
            training_loop.current_loss(),
            training_loop.current_step(),
            training_loop.total_tokens(),
        )
    } else {
        // Standard LoRA path - full precision base weights with dynamic architecture detection
        tracing::info!("Initializing LoRA model with auto-detected architecture...");

        // Detect weight format and use appropriate loader
        let mut model = match WeightFormat::detect(&model_path) {
            Some(WeightFormat::Gguf) => {
                tracing::info!("Detected GGUF format, loading with dequantization...");
                DynamicLoraModel::from_gguf(&model_path, config.lora.clone())?
            }
            _ => {
                // Default to safetensors (HuggingFace format)
                DynamicLoraModel::from_pretrained(&model_path, config.lora.clone())?
            }
        };
        tracing::info!(
            "Loaded {} model with LoRA adapters",
            model.architecture_name()
        );

        // GDN (Gated Delta Networks) training resource guard: models like Qwen3.5
        // use sequential recurrence that creates O(T * n_layers) allocation nodes.
        // Without gradient checkpointing (custom_vjp not yet available in mlx-rs),
        // this exceeds Metal's 499K buffer limit at seq_len > 512.
        if model.architecture_name() == "Qwen3Next" && config.training.max_seq_len > 512 {
            tracing::warn!(
                "Qwen3.5 (GDN) training with max_seq_len={} may exceed Metal resource limits (499K buffers). \
                 Recommended: --max-seq-len 512 or lower. Gradient checkpointing for GDN requires \
                 custom_vjp which is not yet available in mlx-rs.",
                config.training.max_seq_len
            );
        }

        tracing::info!(
            "Trainable parameters: {}",
            format_param_count(model.num_trainable_params())
        );

        // Enable gradient checkpointing if requested
        if gradient_checkpointing {
            if model.supports_gradient_checkpointing() {
                model.enable_gradient_checkpointing(gradient_checkpointing_layers);
                tracing::info!(
                    "Gradient checkpointing enabled ({} layers per block)",
                    gradient_checkpointing_layers
                );
            } else {
                tracing::warn!(
                    "Gradient checkpointing requested but not supported by {} architecture. \
                     This feature is currently only supported for Qwen3 models.",
                    model.architecture_name()
                );
            }
        }

        // Create training loop
        let mut training_loop = TrainingLoop::new(training_loop_config);

        // Wire metrics callback into training loop for step-level dispatch
        if let Some(cb) = metrics_callback.take() {
            training_loop.add_callback(cb);
        }

        // Resume from checkpoint if requested
        if resume {
            if let Some((lora_params, metadata)) = checkpoint_manager.load_latest()? {
                tracing::info!("Resuming from checkpoint at step {}", metadata.step);
                model.set_lora_parameters(&lora_params);
                training_loop.set_step(metadata.step);
                training_loop.set_epoch(metadata.epoch);
            } else {
                tracing::info!("No checkpoint found, starting fresh");
            }
        }

        tracing::info!("Starting LoRA training...");

        // Run training loop
        // Priority: packed > fused > standard
        if use_sequence_packing {
            // Sequence packing for 2-5x throughput
            let model = training_loop.run_packed(
                model,
                train_dataset.clone(),
                eval_dataset.clone(),
                Some(&checkpoint_manager),
            )?;

            progress.finish_with_message(format!("{:.4}", training_loop.current_loss()));

            // Save final LoRA weights
            let final_path = PathBuf::from(&output_dir).join("lora_weights.safetensors");
            model.save_lora_weights(&final_path)?;
            tracing::info!("Saved LoRA weights to {:?}", final_path);
        } else if (fused || use_jit_compilation) && config.training.gradient_accumulation_steps == 1
        {
            // Fused training step (combines forward/backward/optimizer)
            // JIT compilation requires the fused training path for compile_with_state
            let model = training_loop.run_compiled(
                model,
                train_dataset,
                eval_dataset,
                Some(&checkpoint_manager),
            )?;

            progress.finish_with_message(format!("{:.4}", training_loop.current_loss()));

            // Save final LoRA weights
            let final_path = PathBuf::from(&output_dir).join("lora_weights.safetensors");
            model.save_lora_weights(&final_path)?;
            tracing::info!("Saved LoRA weights to {:?}", final_path);
        } else if use_metal_fused_optimizer {
            // Metal fused optimizer for maximum throughput
            tracing::info!("Using Metal fused optimizer for training");
            training_loop.run_metal_fused(
                &mut model,
                train_dataset,
                eval_dataset,
                Some(&checkpoint_manager),
            )?;

            progress.finish_with_message(format!("{:.4}", training_loop.current_loss()));

            // Save final LoRA weights
            let final_path = PathBuf::from(&output_dir).join("lora_weights.safetensors");
            model.save_lora_weights(&final_path)?;
            tracing::info!("Saved LoRA weights to {:?}", final_path);
        } else {
            if (fused || use_jit_compilation) && config.training.gradient_accumulation_steps != 1 {
                tracing::warn!(
                    "Fused/JIT training requires gradient_accumulation_steps=1, falling back to standard training"
                );
            }
            training_loop.run(
                &mut model,
                train_dataset,
                eval_dataset,
                Some(&checkpoint_manager),
            )?;

            progress.finish_with_message(format!("{:.4}", training_loop.current_loss()));

            // Save final LoRA weights
            let final_path = PathBuf::from(&output_dir).join("lora_weights.safetensors");
            model.save_lora_weights(&final_path)?;
            tracing::info!("Saved LoRA weights to {:?}", final_path);
        }

        // Recover metrics callback from training loop for finalization
        let mut cbs = training_loop.take_callbacks();
        if !cbs.is_empty() && metrics_callback.is_none() {
            metrics_callback = Some(cbs.remove(0));
        }

        (
            training_loop.current_loss(),
            training_loop.current_step(),
            training_loop.total_tokens(),
        )
    };

    // Finalize metrics callback
    if let Some(ref mut callback) = metrics_callback {
        // Write final epoch metrics
        let mut custom = std::collections::HashMap::new();
        custom.insert("total_tokens".to_string(), total_tokens as f64);
        custom.insert("total_steps".to_string(), final_step as f64);
        callback.on_epoch_end(
            num_epochs.saturating_sub(1),
            &pmetal_core::EvalMetrics {
                loss: final_loss,
                perplexity: final_loss.exp(),
                accuracy: None,
                custom,
            },
        );
        callback.on_train_end();
        if let Some(ref path) = metrics_path_resolved {
            tracing::info!("Metrics saved to {:?}", path);
        }
    }

    println!("\n========================================");
    println!("  Training Complete!");
    println!("========================================");
    println!("Final Loss:   {:.4}", final_loss);
    println!("Total Steps:  {}", final_step);
    println!("Total Tokens: {}", total_tokens);
    println!("Output:       {}", output_dir);
    println!("========================================");

    println!("\nNext steps:");
    println!(
        "  Inference:  pmetal infer -m {} --lora {}/lora_weights.safetensors -p \"Your prompt\"",
        config.model.model_id, output_dir
    );
    println!(
        "  Quantize:   pmetal quantize -m {} --lora {}/lora_weights.safetensors -o model.gguf",
        config.model.model_id, output_dir
    );

    Ok(())
}

/// Run inference with a model.
///
/// Supports any architecture via DynamicModel, with optional LoRA support
/// for Llama models. Uses KV-cached generation for fast inference.
///
/// Implements SOTA sampling: temperature, top-k, top-p, min-p, repetition penalty,
/// frequency penalty, and presence penalty.
///
/// With `--fp8`, weights are quantized to FP8 E4M3 format for ~2x memory savings.
#[allow(clippy::too_many_arguments)]
async fn run_inference(
    model_id: &str,
    lora_path: Option<&str>,
    prompt: &str,
    max_tokens: usize,
    temperature: Option<f32>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    min_p: Option<f32>,
    repetition_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    seed: Option<u64>,
    chat: bool,
    system: Option<&str>,
    no_thinking: bool,
    metal_sampler: bool,
    compiled: bool,
    _stream: bool,
    minimal: bool,
    show_thinking: bool,
    fp8: bool,
    ane: bool,
    ane_max_seq_len: usize,
) -> anyhow::Result<()> {
    #[cfg(not(feature = "ane"))]
    if ane {
        anyhow::bail!("ANE inference requires the 'ane' feature: cargo build --features ane");
    }
    #[cfg(target_os = "macos")]
    use pmetal_models::generate_cached_metal;
    use pmetal_models::{
        DynamicModel, GenerationConfig, GenerationOutput, generate_cached_compiled,
        generate_minimal_async,
    };

    tracing::info!(model = %model_id, "Loading model for inference");

    // Download model if needed (HuggingFace repo ID contains '/')
    let model_path = if model_id.contains('/') && !PathBuf::from(model_id).exists() {
        tracing::info!("Model not found locally, downloading from HuggingFace Hub...");

        // Download all repo files (configs, tokenizer, weights, etc.)
        let path = pmetal_hub::download_model(model_id, None, None).await?;
        tracing::info!("Model downloaded successfully to {:?}", path);
        path
    } else {
        PathBuf::from(model_id)
    };

    // Load tokenizer (with config-aware special token resolution)
    let tokenizer_path = model_path.join("tokenizer.json");
    let tokenizer = if tokenizer_path.exists() {
        Tokenizer::from_model_dir(&model_path)?
    } else {
        anyhow::bail!(
            "Tokenizer not found at {:?}. GGUF models don't bundle a tokenizer — \
             download the source model first with: pmetal download {}",
            tokenizer_path,
            model_id
        );
    };

    // Check if LoRA is requested
    if let Some(lora) = lora_path {
        return run_inference_with_lora(
            &model_path,
            lora,
            &tokenizer,
            prompt,
            max_tokens,
            temperature,
            top_k,
            top_p,
            min_p,
            repetition_penalty,
            frequency_penalty,
            presence_penalty,
            seed,
            chat,
            system,
            no_thinking,
            show_thinking,
        )
        .await;
    }

    // Use DynamicModel for architecture-agnostic inference
    tracing::info!("Loading model with auto-detected architecture...");
    let mut model = DynamicModel::load(&model_path)?;

    tracing::info!(
        "Model loaded successfully (architecture: {})",
        model.architecture()
    );

    // Apply FP8 quantization if requested
    if fp8 {
        tracing::info!("Quantizing model weights to FP8 E4M3 format...");
        model.quantize_fp8()?;
        tracing::info!("FP8 quantization complete (~2x memory reduction)")
    }

    // Auto-detect if chat mode should be enabled for instruction-tuned models
    let is_instruct_model = is_instruction_tuned(&model_path);
    let use_chat = chat || is_instruct_model;

    if is_instruct_model && !chat {
        tracing::info!("Auto-detected instruction-tuned model, enabling chat template");
    }

    // Load sampling defaults from model's generation_config.json
    let defaults = load_sampling_defaults(&model_path, use_chat && !no_thinking);

    // Apply CLI overrides over model defaults
    let temperature = temperature.unwrap_or(defaults.temperature);
    let top_k = top_k.unwrap_or(defaults.top_k);
    let top_p = top_p.unwrap_or(defaults.top_p);
    let min_p = min_p.unwrap_or(defaults.min_p);
    let repetition_penalty = repetition_penalty.unwrap_or(defaults.repetition_penalty);
    let frequency_penalty = frequency_penalty.unwrap_or(defaults.frequency_penalty);
    let presence_penalty = presence_penalty.unwrap_or(defaults.presence_penalty);

    // Apply chat template if needed
    // The template handles thinking mode - model decides when to think unless --no-thinking
    let (final_prompt, template_type) = if use_chat {
        apply_chat_template(&tokenizer, prompt, system, &model_path, no_thinking)?
    } else {
        (
            prompt.to_string(),
            pmetal_data::chat_templates::ChatTemplateType::ChatMl,
        )
    };

    // Tokenize prompt
    let input_ids = tokenizer.encode(&final_prompt)?;
    tracing::info!("Tokenized {} tokens", input_ids.len());

    // Configure stop tokens — unified collection from all sources
    let stop_tokens = collect_all_stop_tokens(
        &model_path,
        &tokenizer,
        if use_chat { Some(template_type) } else { None },
    );

    // Configure generation with user-specified parameters
    let gen_config = if temperature == 0.0 {
        GenerationConfig::greedy(max_tokens).with_stop_tokens(stop_tokens)
    } else {
        let mut config = GenerationConfig::sampling(max_tokens, temperature)
            .with_top_k(top_k)
            .with_top_p(top_p)
            .with_min_p(min_p)
            .with_repetition_penalty(repetition_penalty)
            .with_frequency_penalty(frequency_penalty)
            .with_presence_penalty(presence_penalty)
            .with_stop_tokens(stop_tokens);

        if let Some(s) = seed {
            config = config.with_seed(s);
        }

        config
    };

    // Print configuration
    println!("\n========================================");
    println!("  PMetal Inference");
    println!("========================================");
    println!("Model:       {}", model_id);
    println!("Temperature: {}", gen_config.temperature);
    println!("Top-k:       {}", gen_config.top_k);
    println!("Top-p:       {}", gen_config.top_p);
    println!("Min-p:       {}", gen_config.min_p);
    println!("Rep penalty: {}", gen_config.repetition_penalty);
    println!("Freq penalty:{}", gen_config.frequency_penalty);
    println!("Pres penalty:{}", gen_config.presence_penalty);
    if let Some(s) = gen_config.seed {
        println!("Seed:        {}", s);
    }
    println!("Max tokens:  {}", max_tokens);
    if use_chat && no_thinking {
        println!("Thinking:    disabled");
    }
    println!("========================================\n");

    println!("Prompt: {}\n", prompt);
    println!("Generating...\n");

    // Create KV cache for efficient generation
    // Cache size = prompt_len + max_tokens + buffer
    let max_seq_len = input_ids.len() + max_tokens + 64;
    let mut cache = model.create_cache(max_seq_len);
    tracing::info!("Created KV cache for {} tokens", max_seq_len);

    // Create Mamba cache for hybrid models (NemotronH)
    let mut mamba_cache = model.create_mamba_cache();
    if mamba_cache.is_some() {
        tracing::info!("Created Mamba cache for hybrid model");
    }

    // Generate with KV cache (and Mamba cache for hybrid models)
    let start = std::time::Instant::now();
    let mut already_streamed = false;

    #[cfg(target_os = "macos")]
    let output = {
        // ANE branch: separate engine with its own weight loading and KV cache
        // Validates architecture compatibility before attempting compilation.
        #[cfg(feature = "ane")]
        let ane_output: Option<GenerationOutput> = if ane {
            // Validate architecture before attempting ANE (saves ~7s on incompatible models)
            let ane_compatible = match std::fs::read_to_string(model_path.join("config.json")) {
                Ok(config_text) => match serde_json::from_str::<serde_json::Value>(&config_text) {
                    Ok(config_json) => {
                        use pmetal_trainer::DynamicAneTrainerConfig;
                        match DynamicAneTrainerConfig::is_ane_compatible(&config_json) {
                            Ok(()) => true,
                            Err(reason) => {
                                tracing::info!("Skipping ANE inference: {}", reason);
                                false
                            }
                        }
                    }
                    Err(_) => true, // Can't parse config, let ANE try anyway
                },
                Err(_) => true, // No config.json, let ANE try anyway
            };

            if ane_compatible {
                tracing::info!(
                    "Attempting ANE-hybrid inference engine (Prefill: ANE, Decode: CPU/vDSP)"
                );
                match pmetal_models::generate_cached_ane(
                    &model_path,
                    &input_ids,
                    &gen_config,
                    ane_max_seq_len,
                ) {
                    Ok(output) => Some(output),
                    Err(e) => {
                        tracing::warn!("ANE inference failed ({}), falling back to GPU", e);
                        None
                    }
                }
            } else {
                None
            }
        } else {
            None
        };
        #[cfg(not(feature = "ane"))]
        let ane_output: Option<GenerationOutput> = None;

        // CPU hybrid engine: try for Qwen3.5 non-MoE when ANE is not compatible
        #[cfg(feature = "ane")]
        let cpu_hybrid_output: Option<GenerationOutput> = if ane_output.is_none() && ane {
            match std::fs::read_to_string(model_path.join("config.json")) {
                Ok(config_text) => match serde_json::from_str::<serde_json::Value>(&config_text) {
                    Ok(config_json) => {
                        use pmetal_models::is_hybrid_cpu_compatible;
                        match is_hybrid_cpu_compatible(&config_json) {
                            Ok(()) => {
                                tracing::info!(
                                    "Attempting CPU GEMV hybrid engine (Qwen3.5 decode: CPU/vDSP)"
                                );
                                match pmetal_models::generate_cached_hybrid_cpu(
                                    &model_path,
                                    &input_ids,
                                    &gen_config,
                                ) {
                                    Ok(output) => Some(output),
                                    Err(e) => {
                                        tracing::warn!(
                                            "CPU hybrid engine failed ({}), falling back to GPU",
                                            e
                                        );
                                        None
                                    }
                                }
                            }
                            Err(_) => None,
                        }
                    }
                    Err(_) => None,
                },
                Err(_) => None,
            }
        } else {
            None
        };
        #[cfg(not(feature = "ane"))]
        let cpu_hybrid_output: Option<GenerationOutput> = None;

        if let Some(output) = ane_output {
            output
        } else if let Some(output) = cpu_hybrid_output {
            output
        } else if minimal {
            tracing::info!("Using minimal async generation (debugging)");
            generate_minimal_async(
                |input, cache| {
                    model.forward_with_hybrid_cache(input, None, Some(cache), mamba_cache.as_mut())
                },
                &input_ids,
                gen_config,
                &mut cache,
            )?
        } else if metal_sampler {
            tracing::info!("Using fused Metal sampling kernel");
            generate_cached_metal(
                |input, cache| {
                    model.forward_with_hybrid_cache(input, None, Some(cache), mamba_cache.as_mut())
                },
                &input_ids,
                gen_config,
                &mut cache,
            )?
        } else if compiled {
            tracing::info!("Using JIT-compiled sampling (mlx_lm style)");
            generate_cached_compiled(
                |input, cache| {
                    model.forward_with_hybrid_cache(input, None, Some(cache), mamba_cache.as_mut())
                },
                &input_ids,
                gen_config,
                &mut cache,
            )?
        } else {
            // Default: streaming async generation — tokens printed as they're produced
            already_streamed = true;
            use pmetal_models::generate_cached_async_streaming;
            use std::io::Write;

            let mut token_buf: Vec<u32> = Vec::new();
            let mut streamed_text = String::new();
            let tokenizer_ref = &tokenizer;

            generate_cached_async_streaming(
                |input, cache| {
                    model.forward_with_hybrid_cache(input, None, Some(cache), mamba_cache.as_mut())
                },
                &input_ids,
                gen_config,
                &mut cache,
                |token_id| {
                    token_buf.push(token_id);
                    if let Ok(text) = tokenizer_ref.decode(&token_buf) {
                        if text.len() > streamed_text.len() {
                            let delta = &text[streamed_text.len()..];
                            let _ = std::io::stdout().write_all(delta.as_bytes());
                            let _ = std::io::stdout().flush();
                        }
                        streamed_text = text;
                    }
                    true
                },
            )?
        }
    };

    #[cfg(not(target_os = "macos"))]
    let output = {
        let _ = metal_sampler; // Suppress unused warning
        let _ = ane;
        if minimal {
            tracing::info!("Using minimal async generation (debugging)");
            generate_minimal_async(
                |input, cache| {
                    model.forward_with_hybrid_cache(input, None, Some(cache), mamba_cache.as_mut())
                },
                &input_ids,
                gen_config,
                &mut cache,
            )?
        } else if compiled {
            tracing::info!("Using JIT-compiled sampling (mlx_lm style)");
            generate_cached_compiled(
                |input, cache| {
                    model.forward_with_hybrid_cache(input, None, Some(cache), mamba_cache.as_mut())
                },
                &input_ids,
                gen_config,
                &mut cache,
            )?
        } else {
            // Default: streaming async generation
            already_streamed = true;
            use pmetal_models::generate_cached_async_streaming;
            use std::io::Write;

            let mut token_buf: Vec<u32> = Vec::new();
            let mut streamed_text = String::new();
            let tokenizer_ref = &tokenizer;

            generate_cached_async_streaming(
                |input, cache| {
                    model.forward_with_hybrid_cache(input, None, Some(cache), mamba_cache.as_mut())
                },
                &input_ids,
                gen_config,
                &mut cache,
                |token_id| {
                    token_buf.push(token_id);
                    if let Ok(text) = tokenizer_ref.decode(&token_buf) {
                        if text.len() > streamed_text.len() {
                            let delta = &text[streamed_text.len()..];
                            let _ = std::io::stdout().write_all(delta.as_bytes());
                            let _ = std::io::stdout().flush();
                        }
                        streamed_text = text;
                    }
                    true
                },
            )?
        }
    };
    let elapsed = start.elapsed();

    // For non-streaming paths, decode and print the generated text now
    if !already_streamed {
        let generated_tokens = &output.token_ids[input_ids.len()..];
        let raw_text = tokenizer.decode(generated_tokens)?;
        let text = if use_chat && !no_thinking {
            format!("<think>{}", raw_text)
        } else {
            raw_text
        };
        if use_chat && show_thinking {
            if let Some(thinking) = extract_thinking_content(&text) {
                println!("=== Thinking ===");
                println!("{}", thinking);
                println!("=== Response ===");
            }
            println!("{}", extract_final_response(&text));
        } else if use_chat {
            println!("{}", extract_final_response(&text));
        } else {
            println!("{}", text);
        }
    } else {
        println!(); // finish the streamed line
    }

    println!("---");
    println!(
        "Generated {} tokens in {:.2}s ({:.1} tok/s)",
        output.num_generated,
        elapsed.as_secs_f64(),
        output.num_generated as f64 / elapsed.as_secs_f64()
    );
    if output.stopped_by_token {
        println!("Stopped by: EOS token");
    } else {
        println!("Stopped by: max length");
    }

    Ok(())
}

/// Get EOS token IDs from model's generation_config.json.
///
/// Many models (like Qwen3) have multiple EOS tokens that should all stop generation.
#[allow(dead_code)]
fn get_eos_tokens(model_path: &Path, tokenizer: &Tokenizer) -> Vec<u32> {
    collect_all_stop_tokens(model_path, tokenizer, None)
}

/// Collect all stop tokens from every available source.
///
/// Merges tokens from:
/// 1. `generation_config.json` — the model's declared `eos_token_id` (single or array)
/// 2. Chat template EOS — the template-specific end token (e.g. `<|im_end|>` for ChatML)
/// 3. Tokenizer's `eos_token_id` — resolved from special_tokens_map / heuristics
/// 4. Well-known special tokens — if they exist in the vocabulary as single tokens,
///    they're likely EOS candidates (e.g. `<|im_end|>`, `<|eot_id|>`, `<|endoftext|>`)
///
/// Returns a deduplicated list. This ensures fine-tuned models stop correctly
/// regardless of whether they produce the base model's EOS or the chat EOS.
fn collect_all_stop_tokens(
    model_path: &Path,
    tokenizer: &Tokenizer,
    template_type: Option<pmetal_data::chat_templates::ChatTemplateType>,
) -> Vec<u32> {
    let mut tokens = Vec::new();

    // 1. generation_config.json
    let config_path = model_path.join("generation_config.json");
    if config_path.exists() {
        if let Ok(content) = std::fs::read_to_string(&config_path) {
            if let Ok(config) = serde_json::from_str::<serde_json::Value>(&content) {
                if let Some(eos) = config.get("eos_token_id") {
                    if let Some(arr) = eos.as_array() {
                        for v in arr {
                            if let Some(id) = v.as_u64() {
                                tokens.push(id as u32);
                            }
                        }
                    } else if let Some(id) = eos.as_u64() {
                        tokens.push(id as u32);
                    }
                }
            }
        }
    }

    // 2. Chat template EOS (if template type is known)
    if let Some(tt) = template_type {
        let eos_str = tt.eos_token();
        if let Ok(encoded) = tokenizer.encode(eos_str) {
            if encoded.len() == 1 {
                tokens.push(encoded[0]);
            }
        }
    }

    // 3. Tokenizer's resolved eos_token_id
    if let Some(eos) = tokenizer.eos_token_id() {
        tokens.push(eos);
    }

    // 4. Well-known special tokens — probe the vocabulary for common EOS tokens.
    //    Only add tokens that encode to exactly 1 token (i.e. they're real special tokens,
    //    not subword sequences).
    let candidates = [
        "<|im_end|>",
        "<|eot_id|>",
        "<|eot|>",
        "<|endoftext|>",
        "<|end_of_text|>",
        "<end_of_turn>",
        "<|end|>",
        "<|return|>",
        "<|END_OF_TURN_TOKEN|>",
        "<｜end▁of▁sentence｜>",
        "</s>",
    ];
    for candidate in &candidates {
        if let Ok(encoded) = tokenizer.encode(candidate) {
            if encoded.len() == 1 {
                tokens.push(encoded[0]);
            }
        }
    }

    // Deduplicate
    tokens.sort_unstable();
    tokens.dedup();

    // Final fallback
    if tokens.is_empty() {
        tokens.push(2);
    }

    tracing::debug!("Collected stop tokens: {:?}", tokens);
    tokens
}

/// Sampling hyperparameter defaults loaded from model config.
struct SamplingDefaults {
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
    repetition_penalty: f32,
    frequency_penalty: f32,
    presence_penalty: f32,
}

impl Default for SamplingDefaults {
    fn default() -> Self {
        // Qwen3 recommended defaults for non-thinking mode
        Self {
            temperature: 0.7,
            top_k: 20,
            top_p: 0.8,
            min_p: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
        }
    }
}

/// Load sampling defaults from model's generation_config.json.
///
/// Uses Qwen3 recommended values as fallback:
/// - Thinking mode: temp=0.6, top_p=0.95, top_k=20, min_p=0
/// - Non-thinking mode: temp=0.7, top_p=0.8, top_k=20, min_p=0
fn load_sampling_defaults(model_path: &Path, thinking_mode: bool) -> SamplingDefaults {
    // Start with mode-appropriate defaults
    let mut defaults = if thinking_mode {
        SamplingDefaults {
            temperature: 0.6,
            top_k: 20,
            top_p: 0.95,
            min_p: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
        }
    } else {
        SamplingDefaults::default()
    };

    // Try to load from generation_config.json
    let config_path = model_path.join("generation_config.json");
    if config_path.exists() {
        if let Ok(content) = std::fs::read_to_string(&config_path) {
            if let Ok(config) = serde_json::from_str::<serde_json::Value>(&content) {
                // Load each parameter if present
                if let Some(v) = config.get("temperature").and_then(|v| v.as_f64()) {
                    defaults.temperature = v as f32;
                }
                if let Some(v) = config.get("top_k").and_then(|v| v.as_u64()) {
                    defaults.top_k = v as usize;
                }
                if let Some(v) = config.get("top_p").and_then(|v| v.as_f64()) {
                    defaults.top_p = v as f32;
                }
                if let Some(v) = config.get("min_p").and_then(|v| v.as_f64()) {
                    defaults.min_p = v as f32;
                }
                if let Some(v) = config.get("repetition_penalty").and_then(|v| v.as_f64()) {
                    defaults.repetition_penalty = v as f32;
                }
                if let Some(v) = config.get("frequency_penalty").and_then(|v| v.as_f64()) {
                    defaults.frequency_penalty = v as f32;
                }
                if let Some(v) = config.get("presence_penalty").and_then(|v| v.as_f64()) {
                    defaults.presence_penalty = v as f32;
                }
            }
        }
    }

    defaults
}

/// Check if a model is instruction-tuned based on its configuration.
///
/// Looks for indicators like:
/// - chat_template in tokenizer_config.json
/// - "instruct", "chat", "it" in model name
/// - Known instruction-tuned model architectures
fn is_instruction_tuned(model_path: &Path) -> bool {
    // Primary: check for chat_template in tokenizer_config.json (authoritative)
    let config_path = model_path.join("tokenizer_config.json");
    if config_path.exists() {
        if let Ok(content) = std::fs::read_to_string(&config_path) {
            if let Ok(config) = serde_json::from_str::<serde_json::Value>(&content) {
                if config.get("chat_template").is_some() {
                    return true;
                }
            }
        }
    }

    // Fallback: only explicit instruct markers in model name
    let path_str = model_path.to_string_lossy().to_lowercase();
    path_str.contains("instruct")
        || path_str.contains("-it-")
        || path_str.contains("-it/")
        || path_str.ends_with("-it")
        || path_str.contains("chat")
}

/// Apply chat template to the prompt using unified detection.
///
/// Returns the formatted prompt string and the detected `ChatTemplateType` so the caller
/// can select the right stop tokens.
///
/// If `no_thinking` is true, prefills empty thinking block to disable reasoning (ChatML/Phi4 only).
/// Otherwise, the model decides when to use thinking based on query complexity.
fn apply_chat_template(
    _tokenizer: &Tokenizer,
    user_message: &str,
    system_message: Option<&str>,
    model_path: &Path,
    no_thinking: bool,
) -> anyhow::Result<(String, pmetal_data::chat_templates::ChatTemplateType)> {
    use pmetal_data::chat_templates::ChatTemplateType;

    let detected = pmetal_data::chat_templates::detect_chat_template(
        model_path,
        &model_path.to_string_lossy(),
    );

    let formatted = match detected.template_type {
        ChatTemplateType::ChatMl | ChatTemplateType::Qwen => {
            format_chatml(user_message, system_message, no_thinking)
        }
        ChatTemplateType::Llama3 => format_llama3(user_message, system_message),
        ChatTemplateType::Llama2 => format_llama2_inference(user_message, system_message),
        ChatTemplateType::Gemma => format_gemma_inference(user_message, system_message),
        ChatTemplateType::Mistral => format_mistral_inference(user_message, system_message),
        ChatTemplateType::Phi3 => format_phi3_inference(user_message, system_message),
        ChatTemplateType::Phi4 => format_phi4_inference(user_message, system_message, no_thinking),
        ChatTemplateType::GptOss => format_gpt_oss_inference(user_message, system_message),
        ChatTemplateType::Llama4 => format_llama4_inference(user_message, system_message),
        ChatTemplateType::DeepSeek => {
            format_deepseek_inference(user_message, system_message, no_thinking)
        }
        ChatTemplateType::Cohere => format_cohere_inference(user_message, system_message),
        // Alpaca, Vicuna, Zephyr, Custom — fall back to ChatML for inference
        _ => format_chatml(user_message, system_message, no_thinking),
    };

    Ok((formatted, detected.template_type))
}

/// Format message using ChatML template (used by Qwen, many others).
fn format_chatml(user_message: &str, system_message: Option<&str>, no_thinking: bool) -> String {
    format_qwen3_chatml(user_message, system_message, no_thinking)
}

/// Format message using Qwen3 ChatML template.
///
/// By default, the model decides when to use `<think>` blocks based on query complexity.
/// If `no_thinking` is true, prefills empty `<think></think>` to force non-thinking mode.
fn format_qwen3_chatml(
    user_message: &str,
    system_message: Option<&str>,
    no_thinking: bool,
) -> String {
    let mut result = String::new();

    // Always include system block (can be empty per NemotronH template)
    result.push_str("<|im_start|>system\n");
    if let Some(sys) = system_message {
        result.push_str(sys);
    }
    result.push_str("<|im_end|>\n");

    result.push_str("<|im_start|>user\n");
    result.push_str(user_message);
    result.push_str("<|im_end|>\n");
    result.push_str("<|im_start|>assistant\n");

    if no_thinking {
        // Force non-thinking: prefill empty think block without newlines
        // This matches NemotronH's expected format
        result.push_str("<think></think>");
    } else {
        // Prefill <think> to ensure clean thinking output
        // Without this, model sometimes generates </think> first or skips thinking
        result.push_str("<think>\n");
    }

    result
}

/// Format message using Llama 3 template.
fn format_llama3(user_message: &str, system_message: Option<&str>) -> String {
    let mut result = String::from("<|begin_of_text|>");

    if let Some(sys) = system_message {
        result.push_str("<|start_header_id|>system<|end_header_id|>\n\n");
        result.push_str(sys);
        result.push_str("<|eot_id|>");
    }

    result.push_str("<|start_header_id|>user<|end_header_id|>\n\n");
    result.push_str(user_message);
    result.push_str("<|eot_id|>");
    result.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");

    result
}

/// Format message using Llama-2 template for inference.
fn format_llama2_inference(user_message: &str, system_message: Option<&str>) -> String {
    let mut result = String::from("<s>[INST] ");
    if let Some(sys) = system_message {
        result.push_str(&format!("<<SYS>>\n{}\n<</SYS>>\n\n", sys));
    }
    result.push_str(user_message);
    result.push_str(" [/INST] ");
    result
}

/// Format message using Gemma template for inference.
fn format_gemma_inference(user_message: &str, system_message: Option<&str>) -> String {
    let mut result = String::new();
    if let Some(sys) = system_message {
        // Gemma folds system into a user turn
        result.push_str(&format!(
            "<start_of_turn>user\n{}\n\n{}<end_of_turn>\n",
            sys, user_message
        ));
    } else {
        result.push_str(&format!(
            "<start_of_turn>user\n{}<end_of_turn>\n",
            user_message
        ));
    }
    result.push_str("<start_of_turn>model\n");
    result
}

/// Format message using Mistral template for inference.
fn format_mistral_inference(user_message: &str, system_message: Option<&str>) -> String {
    let mut result = String::from("[INST] ");
    if let Some(sys) = system_message {
        result.push_str(sys);
        result.push_str("\n\n");
    }
    result.push_str(user_message);
    result.push_str(" [/INST]");
    result
}

/// Format message using Phi-3 template for inference.
fn format_phi3_inference(user_message: &str, system_message: Option<&str>) -> String {
    let mut result = String::new();
    if let Some(sys) = system_message {
        result.push_str("<|system|>\n");
        result.push_str(sys);
        result.push_str("<|end|>\n");
    }
    result.push_str("<|user|>\n");
    result.push_str(user_message);
    result.push_str("<|end|>\n");
    result.push_str("<|assistant|>\n");
    result
}

/// Format message using Phi-4 template for inference.
///
/// Phi-4 uses `<|im_sep|>` instead of the newline separator in standard ChatML.
fn format_phi4_inference(
    user_message: &str,
    system_message: Option<&str>,
    no_thinking: bool,
) -> String {
    let mut result = String::new();

    if let Some(sys) = system_message {
        result.push_str("<|im_start|>system<|im_sep|>");
        result.push_str(sys);
        result.push_str("<|im_end|>");
    }

    result.push_str("<|im_start|>user<|im_sep|>");
    result.push_str(user_message);
    result.push_str("<|im_end|>");
    result.push_str("<|im_start|>assistant<|im_sep|>");

    if no_thinking {
        result.push_str("<think></think>");
    }

    result
}

/// Format message using GPT-OSS Harmony template for inference.
fn format_gpt_oss_inference(user_message: &str, system_message: Option<&str>) -> String {
    let mut result = String::new();
    if let Some(sys) = system_message {
        result.push_str("<|start|>system<|message|>");
        result.push_str(sys);
        result.push_str("<|end|>");
    }
    result.push_str("<|start|>user<|message|>");
    result.push_str(user_message);
    result.push_str("<|end|>");
    result.push_str("<|start|>assistant<|channel|>final<|message|>");
    result
}

/// Format message using Llama 4 template for inference.
///
/// Llama 4 uses `<|header_start|>`/`<|header_end|>` and `<|eot|>` (not Llama 3's tokens).
fn format_llama4_inference(user_message: &str, system_message: Option<&str>) -> String {
    let mut result = String::from("<|begin_of_text|>");

    if let Some(sys) = system_message {
        result.push_str("<|header_start|>system<|header_end|>\n\n");
        result.push_str(sys);
        result.push_str("<|eot|>");
    }

    result.push_str("<|header_start|>user<|header_end|>\n\n");
    result.push_str(user_message);
    result.push_str("<|eot|>");
    result.push_str("<|header_start|>assistant<|header_end|>\n\n");

    result
}

/// Format message using DeepSeek template for inference.
///
/// Uses full-width unicode characters in token names.
/// V3.1+ supports thinking mode via `<think>` / `</think>` prefill.
fn format_deepseek_inference(
    user_message: &str,
    system_message: Option<&str>,
    no_thinking: bool,
) -> String {
    let mut result = String::from("<｜begin▁of▁sentence｜>");

    if let Some(sys) = system_message {
        result.push_str(sys);
    }

    result.push_str("<｜User｜>");
    result.push_str(user_message);
    result.push_str("<｜Assistant｜>");

    if no_thinking {
        result.push_str("</think>");
    } else {
        result.push_str("<think>\n");
    }

    result
}

/// Format message using Cohere Command R template for inference.
fn format_cohere_inference(user_message: &str, system_message: Option<&str>) -> String {
    let mut result = String::from("<BOS_TOKEN>");

    if let Some(sys) = system_message {
        result.push_str("<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>");
        result.push_str(sys);
        result.push_str("<|END_OF_TURN_TOKEN|>");
    }

    result.push_str("<|START_OF_TURN_TOKEN|><|USER_TOKEN|>");
    result.push_str(user_message);
    result.push_str("<|END_OF_TURN_TOKEN|>");
    result.push_str("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>");

    result
}

/// Get stop tokens appropriate for a given chat template type.
///
/// Encodes the template's EOS token via the tokenizer; falls back to the generic
/// `get_eos_tokens` if encoding fails.
#[allow(dead_code)]
fn get_chat_stop_tokens(
    template_type: pmetal_data::chat_templates::ChatTemplateType,
    tokenizer: &Tokenizer,
) -> Vec<u32> {
    // Delegate to the unified collector — merges generation_config.json EOS,
    // chat template EOS, tokenizer EOS, and well-known special tokens.
    // This ensures that chat-mode inference on a base model with LoRA
    // still stops on both <|im_end|> AND <|endoftext|>.
    //
    // NOTE: We need model_path here. Since we don't have it in scope,
    // reconstruct from tokenizer's directory (callers always loaded the tokenizer
    // from model_path). For now, use the old approach enhanced with the
    // well-known tokens probe.
    let eos_str = template_type.eos_token();
    let mut tokens = Vec::new();

    // Template-specific EOS
    if let Ok(encoded) = tokenizer.encode(eos_str) {
        if encoded.len() == 1 {
            tokens.push(encoded[0]);
        }
    }

    // Hardcoded fallbacks for common models
    if tokens.is_empty() {
        match template_type {
            pmetal_data::chat_templates::ChatTemplateType::ChatMl
            | pmetal_data::chat_templates::ChatTemplateType::Qwen
            | pmetal_data::chat_templates::ChatTemplateType::Phi4 => {
                tokens.push(151645); // <|im_end|>
            }
            pmetal_data::chat_templates::ChatTemplateType::Llama3 => {
                tokens.push(128009); // <|eot_id|>
            }
            _ => {
                if let Ok(encoded) = tokenizer.encode("</s>") {
                    if encoded.len() == 1 {
                        tokens.push(encoded[0]);
                    }
                }
                if tokens.is_empty() {
                    tokens.push(2);
                }
            }
        }
    }

    // Also include the tokenizer's native EOS — critical for base models
    // fine-tuned with LoRA that might emit either the chat EOS or the base EOS.
    if let Some(eos) = tokenizer.eos_token_id() {
        if !tokens.contains(&eos) {
            tokens.push(eos);
        }
    }

    // Probe well-known special tokens in vocabulary
    let candidates = [
        "<|im_end|>",
        "<|eot_id|>",
        "<|eot|>",
        "<|endoftext|>",
        "<|end_of_text|>",
        "<end_of_turn>",
        "<|end|>",
        "<|return|>",
        "<|END_OF_TURN_TOKEN|>",
        "<｜end▁of▁sentence｜>",
        "</s>",
    ];
    for candidate in &candidates {
        if let Ok(encoded) = tokenizer.encode(candidate) {
            if encoded.len() == 1 && !tokens.contains(&encoded[0]) {
                tokens.push(encoded[0]);
            }
        }
    }

    tokens
}

/// Extract the final response after </think> tag, discarding thinking content.
///
/// Handles several cases:
/// 1. Complete thinking: `<think>...</think>response` -> returns `response`
/// 2. Incomplete thinking (hit max tokens): `<think>...` -> returns empty (model didn't finish)
/// 3. No thinking: `response` -> returns `response`
fn extract_final_response(text: &str) -> String {
    // Case 1: Find complete </think> tag
    if let Some(pos) = text.rfind("</think>") {
        let after_think = &text[pos + "</think>".len()..];
        // Clean up any stray <think> tags (small models sometimes output malformed content)
        let cleaned = after_think
            .trim()
            .trim_start_matches("<think>")
            .trim_start_matches('\n');
        return strip_eos_tokens(cleaned).to_string();
    }

    // Case 2: Incomplete thinking - model started <think> but never finished
    // Since there's no </think>, the model was still thinking when it hit max tokens
    if text.contains("<think>") {
        return "[Response truncated - model was still thinking. Use --no-thinking or increase --max-tokens]".to_string();
    }

    // Case 3: No thinking block, return as-is
    strip_eos_tokens(text).to_string()
}

/// Strip any known EOS / stop tokens from the end of generated text.
fn strip_eos_tokens(text: &str) -> &str {
    // Order: longest tokens first to avoid partial matches
    const EOS_TOKENS: &[&str] = &[
        "<|endoftext|>",
        "<|im_end|>",
        "<|eot_id|>",
        "<|eot|>",
        "<end_of_turn>",
        "<|END_OF_TURN_TOKEN|>",
        "<｜end▁of▁sentence｜>",
        "<|return|>",
        "<|end|>",
        "</s>",
    ];

    let mut s = text.trim();
    // Loop in case multiple EOS tokens are concatenated
    loop {
        let before = s;
        for tok in EOS_TOKENS {
            s = s.trim_end_matches(tok).trim();
        }
        if s == before {
            break;
        }
    }
    s
}

/// Extract thinking content from response (for display purposes).
///
/// Handles cases where the model generates multiple `<think>` tokens at the start
/// by finding the last complete `<think>...</think>` block.
fn extract_thinking_content(text: &str) -> Option<String> {
    // Find the closing </think> tag first
    let end = text.rfind("</think>")?;

    // Find the last <think> tag before </think> that starts actual content
    // (skip repeated <think> tags at the start)
    let search_region = &text[..end];

    // Find the last <think> that's followed by actual text content, not just more <think> tags
    let mut last_real_start = None;
    let mut pos = 0;
    while let Some(start) = search_region[pos..].find("<think>") {
        let absolute_start = pos + start;
        let after_tag = &search_region[absolute_start + "<think>".len()..];

        // Check if this is followed by real content (not just another <think> or whitespace then <think>)
        let trimmed = after_tag.trim_start();
        if !trimmed.starts_with("<think>") && !trimmed.is_empty() {
            last_real_start = Some(absolute_start);
        }

        pos = absolute_start + "<think>".len();
    }

    if let Some(start) = last_real_start {
        let thinking = &text[start + "<think>".len()..end];
        // Clean up the thinking content
        let cleaned = thinking
            .trim()
            .trim_start_matches("<think>")
            .trim_start_matches('\n')
            .trim();
        if !cleaned.is_empty() {
            return Some(cleaned.to_string());
        }
    }

    // Fallback: simple extraction if the above didn't work
    if let Some(start) = text.find("<think>") {
        if end > start {
            let thinking = &text[start + "<think>".len()..end];
            let cleaned = thinking.trim();
            if !cleaned.is_empty() {
                return Some(cleaned.to_string());
            }
        }
    }

    None
}

/// Run inference with LoRA adapter (supports all architectures via DynamicLoraModel).
///
/// Mirrors the main inference path: auto-detects chat mode, applies chat template,
/// configures stop tokens (including chat EOS), and respects all sampling parameters.
#[allow(clippy::too_many_arguments)]
async fn run_inference_with_lora(
    model_path: &Path,
    lora_path: &str,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_tokens: usize,
    temperature: Option<f32>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    min_p: Option<f32>,
    repetition_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    seed: Option<u64>,
    chat: bool,
    system: Option<&str>,
    no_thinking: bool,
    show_thinking: bool,
) -> anyhow::Result<()> {
    use pmetal_core::LoraConfig;
    use pmetal_lora::{DynamicLoraModel, TrainableModel};
    use pmetal_models::{GenerationConfig, generate_cached_async_streaming};

    // Create LoRA config - we'll load actual weights which override this
    let lora_config = LoraConfig {
        r: 16, // Will be overridden by loaded weights
        alpha: 16.0,
        ..Default::default()
    };

    // Use DynamicLoraModel for automatic architecture detection
    tracing::info!("Loading model with auto-detected architecture...");
    let mut model = DynamicLoraModel::from_pretrained(model_path, lora_config)?;
    tracing::info!("Detected architecture: {}", model.architecture_name());

    // Load LoRA adapter weights
    tracing::info!("Loading LoRA adapter from {:?}...", lora_path);
    model.load_lora_weights(lora_path)?;

    tracing::info!("Model loaded successfully");

    // Auto-detect chat mode: LoRA fine-tuned models almost always use chat format,
    // so enable chat if the model has chat special tokens even if it's a "base" model.
    let is_instruct = is_instruction_tuned(model_path);
    let has_chat_tokens = tokenizer
        .encode("<|im_end|>")
        .is_ok_and(|t| t.len() == 1)
        || tokenizer
            .encode("<|eot_id|>")
            .is_ok_and(|t| t.len() == 1);
    let use_chat = chat || is_instruct || has_chat_tokens;

    if !chat && use_chat {
        tracing::info!(
            "Auto-enabled chat mode for LoRA inference ({})",
            if is_instruct {
                "instruction-tuned model"
            } else {
                "model has chat special tokens"
            }
        );
    }

    // Load sampling defaults from model's generation_config.json
    let defaults = load_sampling_defaults(model_path, use_chat && !no_thinking);

    // Apply CLI overrides over model defaults
    let temperature = temperature.unwrap_or(defaults.temperature);
    let top_k = top_k.unwrap_or(defaults.top_k);
    let top_p = top_p.unwrap_or(defaults.top_p);
    let min_p = min_p.unwrap_or(defaults.min_p);
    let repetition_penalty = repetition_penalty.unwrap_or(defaults.repetition_penalty);
    let frequency_penalty = frequency_penalty.unwrap_or(defaults.frequency_penalty);
    let presence_penalty = presence_penalty.unwrap_or(defaults.presence_penalty);

    // Apply chat template if using chat mode
    let (final_prompt, template_type) = if use_chat {
        apply_chat_template(tokenizer, prompt, system, model_path, no_thinking)?
    } else {
        (
            prompt.to_string(),
            pmetal_data::chat_templates::ChatTemplateType::ChatMl,
        )
    };

    // Tokenize prompt
    let input_ids = tokenizer.encode(&final_prompt)?;
    tracing::info!("Tokenized {} tokens", input_ids.len());

    // Configure stop tokens — unified collection from all sources
    let stop_tokens = collect_all_stop_tokens(
        model_path,
        tokenizer,
        if use_chat { Some(template_type) } else { None },
    );

    // Configure generation with all sampling parameters
    let gen_config = if temperature == 0.0 {
        GenerationConfig::greedy(max_tokens).with_stop_tokens(stop_tokens)
    } else {
        let mut config = GenerationConfig::sampling(max_tokens, temperature)
            .with_top_k(top_k)
            .with_top_p(top_p)
            .with_min_p(min_p)
            .with_repetition_penalty(repetition_penalty)
            .with_frequency_penalty(frequency_penalty)
            .with_presence_penalty(presence_penalty)
            .with_stop_tokens(stop_tokens);

        if let Some(s) = seed {
            config = config.with_seed(s);
        }

        config
    };

    // Print configuration
    println!("\n========================================");
    println!("  PMetal Inference (LoRA)");
    println!("========================================");
    println!("LoRA:        {}", lora_path);
    println!("Chat mode:   {}", use_chat);
    println!("Temperature: {}", gen_config.temperature);
    println!("Stop tokens: {:?}", gen_config.stop_tokens);
    println!("========================================\n");

    println!("Prompt: {}\n", prompt);
    println!("Generating with KV cache...\n");

    // Create KV cache for efficient generation
    let max_seq_len = input_ids.len() + max_tokens + 64;
    let mut cache = model
        .create_cache(max_seq_len)
        .ok_or_else(|| anyhow::anyhow!("Model does not support KV cache"))?;
    tracing::info!("Created KV cache for {} tokens", max_seq_len);

    // Create Mamba cache for hybrid models (Qwen3.5 GDN layers)
    let mut mamba_cache = model.create_mamba_cache();
    if mamba_cache.is_some() {
        tracing::info!("Created Mamba cache for hybrid LoRA model");
    }

    let start = std::time::Instant::now();

    // Generate with hybrid cache — stream tokens as they're produced
    let mut token_buf: Vec<u32> = Vec::new();
    let mut streamed_text = String::new();
    let tokenizer_ref = tokenizer;

    let output = generate_cached_async_streaming(
        |input, cache| {
            model
                .forward_with_hybrid_cache(input, None, Some(cache), mamba_cache.as_mut())
                .map_err(|e| mlx_rs::error::Exception::custom(e.to_string()))
        },
        &input_ids,
        gen_config,
        &mut cache,
        |token_id| {
            use std::io::Write;
            token_buf.push(token_id);
            if let Ok(text) = tokenizer_ref.decode(&token_buf) {
                if text.len() > streamed_text.len() {
                    let delta = &text[streamed_text.len()..];
                    let _ = std::io::stdout().write_all(delta.as_bytes());
                    let _ = std::io::stdout().flush();
                }
                streamed_text = text;
            }
            true
        },
    )?;

    let elapsed = start.elapsed();
    println!(); // finish streamed line

    // Post-process: extract thinking and clean up EOS tokens
    if use_chat && show_thinking {
        if let Some(thinking) = extract_thinking_content(&streamed_text) {
            if !thinking.is_empty() {
                println!("\n--- Thinking ---");
                println!("{}", thinking.trim());
                println!("--- End Thinking ---\n");
            }
        }
    }

    let tokens_per_sec = output.num_generated as f64 / elapsed.as_secs_f64();
    println!("\n---");
    println!(
        "Generated {} tokens in {:.2}s ({:.1} tok/s)",
        output.num_generated,
        elapsed.as_secs_f64(),
        tokens_per_sec
    );
    if output.stopped_by_token {
        println!("Stopped by: EOS token");
    } else {
        println!("Stopped by: max length");
    }

    Ok(())
}

/// Run Ollama subcommands.
async fn run_ollama_command(action: OllamaAction) -> anyhow::Result<()> {
    match action {
        OllamaAction::Modelfile {
            base,
            lora,
            output,
            system,
            temperature,
            num_ctx,
            top_k,
            top_p,
            template,
            license,
        } => {
            generate_modelfile(
                &base,
                lora.as_deref(),
                &output,
                system.as_deref(),
                temperature,
                num_ctx,
                top_k,
                top_p,
                template,
                license.as_deref(),
            )?;
        }

        OllamaAction::Create {
            name,
            base,
            lora,
            system,
            temperature,
            num_ctx,
            template,
        } => {
            create_ollama_model(
                &name,
                &base,
                lora.as_deref(),
                system.as_deref(),
                temperature,
                num_ctx,
                template,
            )?;
        }

        OllamaAction::Templates => {
            print_ollama_templates();
        }
    }

    Ok(())
}

/// Generate a Modelfile for Ollama.
fn generate_modelfile(
    base: &str,
    lora: Option<&str>,
    output: &str,
    system: Option<&str>,
    temperature: Option<f32>,
    num_ctx: Option<i32>,
    top_k: Option<i32>,
    top_p: Option<f32>,
    template: Option<OllamaTemplate>,
    license: Option<&str>,
) -> anyhow::Result<()> {
    // Validate output path to prevent path traversal
    let output_path = validate_file_path(output, true)?;

    println!("========================================");
    println!("  PMetal Ollama Export");
    println!("========================================");
    println!("Base Model:  {}", base);
    if let Some(lora_path) = lora {
        println!("LoRA:        {}", lora_path);
    }
    println!("Output:      {}", output_path.display());
    println!("========================================\n");

    // Build Modelfile
    let mut builder = ModelfileBuilder::new().from(base);

    // Add LoRA adapter if specified
    if let Some(lora_path) = lora {
        builder = builder.adapter(lora_path);
    }

    // Add system prompt
    if let Some(sys) = system {
        builder = builder.system(sys);
    }

    // Add parameters
    if let Some(temp) = temperature {
        builder = builder.temperature(temp);
    }
    if let Some(ctx) = num_ctx {
        builder = builder.num_ctx(ctx);
    }
    if let Some(k) = top_k {
        builder = builder.top_k(k);
    }
    if let Some(p) = top_p {
        builder = builder.top_p(p);
    }

    // Add template
    if let Some(tmpl) = template {
        let template_str = get_ollama_template(tmpl);
        builder = builder.template(template_str);
    } else {
        // Try to auto-detect template from base model name
        if let Some(detected_template) = detect_template_from_model(base) {
            builder = builder.template(detected_template);
            println!("Auto-detected template from model name");
        }
    }

    // Add license
    if let Some(lic) = license {
        builder = builder.license(lic);
    }

    // Build and write
    builder.write_to_file(&output_path)?;
    println!("Modelfile written to: {}", output_path.display());

    println!("\nTo create the model in Ollama, run:");
    println!("  ollama create <model-name> -f {}", output_path.display());

    Ok(())
}

/// Validate model name for Ollama (prevent command injection).
fn validate_ollama_model_name(name: &str) -> anyhow::Result<()> {
    // Allow alphanumeric, hyphen, underscore, period, forward slash (for namespaces)
    // Reject anything that could be interpreted as shell metacharacters
    if name.is_empty() {
        anyhow::bail!("Model name cannot be empty");
    }
    if name.len() > 255 {
        anyhow::bail!("Model name too long (max 255 characters)");
    }
    if name.starts_with('.') || name.starts_with('-') {
        anyhow::bail!("Model name cannot start with '.' or '-'");
    }
    if !name
        .chars()
        .all(|c| c.is_alphanumeric() || matches!(c, '-' | '_' | '.' | '/' | ':'))
    {
        anyhow::bail!(
            "Invalid model name '{}'. Name must contain only alphanumeric characters, \
             hyphens, underscores, periods, colons, and forward slashes.",
            name
        );
    }
    Ok(())
}

/// Validate file path (prevent path traversal).
fn validate_file_path(path: &str, allow_creation: bool) -> anyhow::Result<std::path::PathBuf> {
    let path = std::path::Path::new(path);

    // Prevent path traversal
    if path
        .components()
        .any(|c| matches!(c, std::path::Component::ParentDir))
    {
        anyhow::bail!("Invalid path: path traversal detected (.. not allowed)");
    }

    // Get canonical path
    let canonical = if path.exists() {
        path.canonicalize()?
    } else if allow_creation {
        // If file doesn't exist yet, canonicalize parent
        if let Some(parent) = path.parent() {
            if parent.as_os_str().is_empty() {
                let file_name = path
                    .file_name()
                    .ok_or_else(|| anyhow::anyhow!("Invalid path: no file name"))?;
                std::env::current_dir()?.join(file_name)
            } else {
                let file_name = path
                    .file_name()
                    .ok_or_else(|| anyhow::anyhow!("Invalid path: no file name"))?;
                parent.canonicalize()?.join(file_name)
            }
        } else {
            std::env::current_dir()?.join(path)
        }
    } else {
        anyhow::bail!("Path does not exist: {}", path.display());
    };

    Ok(canonical)
}

/// Create and register a model with Ollama.
fn create_ollama_model(
    name: &str,
    base: &str,
    lora: Option<&str>,
    system: Option<&str>,
    temperature: Option<f32>,
    num_ctx: Option<i32>,
    template: Option<OllamaTemplate>,
) -> anyhow::Result<()> {
    // Validate model name to prevent command injection
    validate_ollama_model_name(name)?;

    // Create secure temporary file (auto-cleaned on drop)
    let modelfile = tempfile::Builder::new()
        .prefix("pmetal-modelfile-")
        .suffix(".txt")
        .tempfile()?;
    let modelfile_path = modelfile.path().to_path_buf();
    let modelfile_str = modelfile_path.to_string_lossy().to_string();

    generate_modelfile(
        base,
        lora,
        &modelfile_str,
        system,
        temperature,
        num_ctx,
        None,
        None,
        template,
        None,
    )?;

    println!("\nCreating Ollama model '{}'...", name);

    // Run ollama create
    let status = std::process::Command::new("ollama")
        .args(["create", name, "-f", &modelfile_str])
        .status();

    match status {
        Ok(exit_status) if exit_status.success() => {
            println!("\nModel '{}' created successfully!", name);
            println!("\nTo use the model, run:");
            println!("  ollama run {}", name);
            // modelfile is auto-cleaned on drop
        }
        Ok(exit_status) => {
            // Persist the temp file so user can inspect it
            let persisted = modelfile.into_temp_path();
            let kept_path = persisted.keep()?;
            anyhow::bail!(
                "ollama create failed with exit code: {:?}. \
                 Modelfile saved at: {}",
                exit_status.code(),
                kept_path.display()
            );
        }
        Err(e) => {
            if e.kind() == std::io::ErrorKind::NotFound {
                let persisted = modelfile.into_temp_path();
                let kept_path = persisted.keep()?;
                println!("\nOllama not found. Please install Ollama first:");
                println!("  https://ollama.ai/download");
                println!("\nModelfile has been saved to: {}", kept_path.display());
                println!("Once Ollama is installed, run:");
                println!("  ollama create {} -f {}", name, kept_path.display());
            } else {
                anyhow::bail!("Failed to run ollama: {}", e);
            }
        }
    }

    Ok(())
}

/// Print available Ollama templates.
fn print_ollama_templates() {
    println!("Available Ollama Templates:");
    println!("========================================\n");

    println!("llama3 - Llama 3 Chat Format");
    println!("  Uses: <|start_header_id|>...<|end_header_id|> format");
    println!("  Best for: Llama 3, Llama 3.1, Llama 3.2, Llama 4\n");

    println!("qwen3 - Qwen3/ChatML Format");
    println!("  Uses: <|im_start|>...<|im_end|> format");
    println!("  Best for: Qwen 2, Qwen 2.5, Qwen 3\n");

    println!("gemma - Gemma Instruct Format");
    println!("  Uses: <start_of_turn>...<end_of_turn> format");
    println!("  Best for: Gemma 2, Gemma 3\n");

    println!("mistral - Mistral Instruct Format");
    println!("  Uses: [INST]...[/INST] format");
    println!("  Best for: Mistral, Mixtral\n");

    println!("phi3 - Phi-3 Instruct Format");
    println!("  Uses: <|system|>...<|end|> format");
    println!("  Best for: Phi 3, Phi 4\n");

    println!("deepseek - DeepSeek Chat Format");
    println!("  Uses: <|begin_of_sentence|>User:...Assistant: format");
    println!("  Best for: DeepSeek, DeepSeek-V2, DeepSeek-V3\n");

    println!("========================================");
    println!("Usage: pmetal ollama modelfile --base <model> --template <template>");
}

/// Get the Ollama template string for a template type.
fn get_ollama_template(template: OllamaTemplate) -> &'static str {
    match template {
        OllamaTemplate::Llama3 => ollama_templates::LLAMA3_CHAT,
        OllamaTemplate::Qwen3 => ollama_templates::QWEN3_CHAT,
        OllamaTemplate::Gemma => ollama_templates::GEMMA_INSTRUCT,
        OllamaTemplate::Mistral => ollama_templates::MISTRAL_INSTRUCT,
        OllamaTemplate::Phi3 => ollama_templates::PHI3_INSTRUCT,
        OllamaTemplate::DeepSeek => ollama_templates::DEEPSEEK_CHAT,
    }
}

/// Try to detect the appropriate template from the model name.
fn detect_template_from_model(model: &str) -> Option<&'static str> {
    let lower = model.to_lowercase();

    if lower.contains("llama") || lower.contains("meta-llama") {
        Some(ollama_templates::LLAMA3_CHAT)
    } else if lower.contains("qwen") {
        Some(ollama_templates::QWEN3_CHAT)
    } else if lower.contains("gemma") {
        Some(ollama_templates::GEMMA_INSTRUCT)
    } else if lower.contains("mistral") || lower.contains("mixtral") {
        Some(ollama_templates::MISTRAL_INSTRUCT)
    } else if lower.contains("phi") {
        Some(ollama_templates::PHI3_INSTRUCT)
    } else if lower.contains("deepseek") {
        Some(ollama_templates::DEEPSEEK_CHAT)
    } else {
        None
    }
}

/// Run benchmark.
async fn run_benchmark(model: &str, batch_size: usize, seq_len: usize) -> anyhow::Result<()> {
    tracing::info!(
        model = %model,
        batch_size = batch_size,
        seq_len = seq_len,
        "Running benchmark"
    );

    println!("Benchmark Configuration:");
    println!("  Model:      {}", model);
    println!("  Batch Size: {}", batch_size);
    println!("  Seq Length: {}", seq_len);
    println!("\nBenchmarking in progress...");

    // Create dummy config
    let llama_config = LlamaConfig {
        vocab_size: 32000,
        hidden_size: 2048,
        intermediate_size: 5632,
        num_hidden_layers: 22,
        num_attention_heads: 32,
        num_key_value_heads: Some(4),
        max_position_embeddings: 2048,
        ..Default::default()
    };

    let lora_config = LoraConfig {
        r: 16,
        alpha: 32.0,
        ..Default::default()
    };

    let mut model = LlamaLoraForCausalLM::new(llama_config, lora_config)?;

    // Create dummy data
    let input_ids = mlx_rs::Array::zeros::<i32>(&[batch_size as i32, seq_len as i32])?;

    // Warmup
    println!("Warming up...");
    for _ in 0..3 {
        let output = model.forward(&input_ids, None)?;
        output.eval()?;
    }

    // Benchmark
    let iterations = 10;
    let start = std::time::Instant::now();

    for _ in 0..iterations {
        let output = model.forward(&input_ids, None)?;
        output.eval()?;
    }

    let elapsed = start.elapsed();
    let avg_ms = elapsed.as_millis() as f64 / iterations as f64;
    let tokens_per_sec = (batch_size * seq_len) as f64 / (avg_ms / 1000.0);

    println!("\nResults:");
    println!("  Avg Time:       {:.2} ms/iteration", avg_ms);
    println!("  Throughput:     {:.0} tokens/sec", tokens_per_sec);

    let stats = pmetal_mlx::memory::get_memory_stats();
    println!("  Memory Used:    {:.2} GB", stats.used_gb());
    println!("  Peak Memory:    {:.2} GB", stats.peak_gb());

    Ok(())
}

/// Generate a sample configuration file.
fn generate_sample_config(output: &str) -> anyhow::Result<()> {
    let config = FullTrainingConfig {
        model: ModelConfig {
            model_id: "meta-llama/Llama-3.2-1B".to_string(),
            max_seq_len: 2048,
            ..Default::default()
        },
        lora: LoraConfig {
            r: 16,
            alpha: 32.0,
            target_modules: vec![
                "q_proj".to_string(),
                "k_proj".to_string(),
                "v_proj".to_string(),
                "o_proj".to_string(),
            ],
            ..Default::default()
        },
        training: TrainingConfig {
            learning_rate: 2e-4,
            batch_size: 4,
            num_epochs: 3,
            warmup_steps: 100,
            max_seq_len: 2048,
            output_dir: "./output".to_string(),
            logging_steps: 10,
            save_steps: Some(500),
            ..Default::default()
        },
        dataset: DatasetConfig {
            dataset_id: "./data/train.jsonl".to_string(),
            shuffle: true,
            ..Default::default()
        },
    };

    let yaml = serde_yaml::to_string(&config)?;
    std::fs::write(output, yaml)?;

    println!("Sample configuration written to: {}", output);
    println!("\nYou can edit this file and run training with:");
    println!("  pmetal train --config {}", output);

    Ok(())
}

/// Format parameter count with suffix (K, M, B).
fn format_param_count(count: usize) -> String {
    if count >= 1_000_000_000 {
        format!("{:.2}B", count as f64 / 1_000_000_000.0)
    } else if count >= 1_000_000 {
        format!("{:.2}M", count as f64 / 1_000_000.0)
    } else if count >= 1_000 {
        format!("{:.2}K", count as f64 / 1_000.0)
    } else {
        format!("{}", count)
    }
}

/// Validate and sanitize an output path to prevent path traversal attacks.
///
/// This function:
/// 1. Rejects paths containing ".." components
/// 2. Rejects absolute paths that escape the current working directory
/// 3. Canonicalizes paths to resolve symlinks and normalize components
///
/// # Arguments
/// * `path` - The path to validate
/// * `context` - A description of what this path is for (used in error messages)
///
/// # Returns
/// The validated and canonicalized path, or an error if validation fails.
fn validate_output_path(path: &str, context: &str) -> anyhow::Result<PathBuf> {
    let path = PathBuf::from(path);

    // Check for explicit ".." components in the path
    for component in path.components() {
        if matches!(component, Component::ParentDir) {
            anyhow::bail!(
                "Path traversal detected in {}: '{}' contains '..' component. \
                 Please use a path within the current directory.",
                context,
                path.display()
            );
        }
    }

    // Get the current working directory
    let cwd = std::env::current_dir()?;

    // Resolve the path
    let resolved = if path.is_absolute() {
        path.clone()
    } else {
        cwd.join(&path)
    };

    // Canonicalize after creating parent directories if needed
    // For output paths, the directory may not exist yet
    if let Some(parent) = resolved.parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent)?;
        }
    }

    // If the path itself exists, canonicalize it
    // Otherwise, canonicalize the parent and append the filename
    let canonical = if resolved.exists() {
        resolved.canonicalize()?
    } else if let Some(parent) = resolved.parent() {
        let canonical_parent = parent.canonicalize()?;
        if let Some(filename) = resolved.file_name() {
            canonical_parent.join(filename)
        } else {
            canonical_parent
        }
    } else {
        resolved
    };

    // Ensure the canonical path is under the current working directory
    // or is a well-known safe location like /tmp or user home
    let cwd_canonical = cwd.canonicalize()?;
    let home_dir = dirs::home_dir();
    let temp_dir = std::env::temp_dir().canonicalize().ok();

    let is_safe = canonical.starts_with(&cwd_canonical)
        || home_dir
            .as_ref()
            .map(|h| canonical.starts_with(h))
            .unwrap_or(false)
        || temp_dir
            .as_ref()
            .map(|t| canonical.starts_with(t))
            .unwrap_or(false)
        // macOS: /tmp symlinks to /private/tmp, but temp_dir() returns /var/folders/
        || canonical.starts_with("/tmp")
        || canonical.starts_with("/private/tmp");

    if !is_safe {
        anyhow::bail!(
            "Unsafe output path for {}: '{}' resolves to '{}' which is outside \
             the current directory, home directory, and temp directory. \
             Please use a path within a safe location.",
            context,
            path.display(),
            canonical.display()
        );
    }

    Ok(canonical)
}

/// Benchmark FFI overhead to compare Rust mlx-rs vs Python mlx performance.
///
/// Python baseline: ~7420 argmax ops/sec (~0.135ms per op) on Qwen3 vocab size.
fn run_ffi_benchmark() -> anyhow::Result<()> {
    use mlx_rs::{
        Array,
        ops::indexing::{argmax, argmax_axis},
        transforms::eval,
    };
    use std::time::Instant;

    println!("FFI Overhead Benchmark");
    println!("======================\n");

    // Warmup - ensure Metal is ready
    println!("Warming up Metal...");
    let warmup = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]);
    let _ = argmax(&warmup, None)?;
    eval([&warmup])?;

    // Create test array similar to Qwen3 logits
    let vocab_size = 151936; // Qwen3 vocab size
    println!("Creating logits array with vocab_size = {}", vocab_size);
    let logits_data: Vec<f32> = (0..vocab_size).map(|i| (i as f32) * 0.001).collect();
    let logits = Array::from_slice(&logits_data, &[1, vocab_size as i32]);
    eval([&logits])?;

    // Benchmark 1: argmax operations
    let n_iters = 1000;
    println!("\n--- Test 1: argmax({} iterations) ---", n_iters);

    let start = Instant::now();
    for _ in 0..n_iters {
        let result = argmax_axis(&logits, -1, None)?;
        eval([&result])?;
    }
    let elapsed = start.elapsed();

    let per_op_us = elapsed.as_micros() as f64 / n_iters as f64;
    let per_op_ms = per_op_us / 1000.0;
    let ops_per_sec = 1_000_000.0 / per_op_us;

    println!("Total time: {:?}", elapsed);
    println!("Per operation: {:.3} ms ({:.1} us)", per_op_ms, per_op_us);
    println!("Operations per second: {:.0}", ops_per_sec);
    println!("Python baseline: ~7420 ops/sec (~0.135 ms/op)");
    println!("Ratio to Python: {:.2}x", ops_per_sec / 7420.0);

    // Benchmark 2: argmax + reshape (generation loop pattern)
    println!("\n--- Test 2: argmax + reshape (generation pattern) ---");

    let token = Array::from_slice(&[42i32], &[1]);
    eval([&token])?;

    let start = Instant::now();
    for _ in 0..n_iters {
        // Reshape token [1] -> [1, 1] (like our generation loop)
        let input = token.reshape(&[1, 1])?;
        // Simulate logits extraction
        let result = argmax_axis(&logits, -1, None)?;
        eval([&input, &result])?;
    }
    let elapsed = start.elapsed();

    let per_op_us = elapsed.as_micros() as f64 / n_iters as f64;
    let per_op_ms = per_op_us / 1000.0;

    println!("Total time: {:?}", elapsed);
    println!("Per operation: {:.3} ms ({:.1} us)", per_op_ms, per_op_us);
    println!("Equivalent tok/s: {:.0}", 1_000_000.0 / per_op_us);

    // Benchmark 3: Multiple small operations (typical sampling overhead)
    println!("\n--- Test 3: Multiple ops per iteration (sampling simulation) ---");

    let start = Instant::now();
    for _ in 0..n_iters {
        // Simulate temperature scaling
        let scaled = logits.multiply(&Array::from_f32(0.7))?;
        // argmax
        let result = argmax_axis(&scaled, -1, None)?;
        eval([&result])?;
    }
    let elapsed = start.elapsed();

    let per_op_us = elapsed.as_micros() as f64 / n_iters as f64;
    let per_op_ms = per_op_us / 1000.0;

    println!("Total time: {:?}", elapsed);
    println!("Per operation: {:.3} ms ({:.1} us)", per_op_ms, per_op_us);
    println!("Equivalent tok/s: {:.0}", 1_000_000.0 / per_op_us);

    println!("\n======================");
    println!("Analysis:");
    println!("- If Rust matches Python (~0.135ms/op), FFI overhead is NOT the issue");
    println!("- If Rust is significantly slower, we may need direct mlx_sys calls");
    println!("- Compare to actual generation: ~5ms/token means overhead is elsewhere");

    // Test 4: Compare sync vs async eval timing
    println!("\n--- Test 4: async_eval timing ---");

    use mlx_rs::transforms::async_eval;

    let start = Instant::now();
    for _ in 0..n_iters {
        // Build graph (should be fast)
        let scaled = logits.multiply(&Array::from_f32(0.7))?;
        let result = argmax_axis(&scaled, -1, None)?;
        // Schedule async (should be where GPU work happens)
        async_eval([&result])?;
    }
    // Final sync
    eval([&logits])?;
    let elapsed = start.elapsed();

    let per_op_us = elapsed.as_micros() as f64 / n_iters as f64;
    println!(
        "Per iteration with async_eval: {:.3} ms ({:.1} us)",
        per_op_us / 1000.0,
        per_op_us
    );

    // Test 5: Measure graph construction vs execution
    println!("\n--- Test 5: Graph construction vs execution timing ---");

    // Just graph construction, no eval
    let start = Instant::now();
    let mut results = Vec::new();
    for _ in 0..100 {
        let scaled = logits.multiply(&Array::from_f32(0.7))?;
        let result = argmax_axis(&scaled, -1, None)?;
        results.push(result);
    }
    let graph_time = start.elapsed();
    println!("Graph construction (100 iters): {:?}", graph_time);
    println!(
        "Per graph: {:.3} ms",
        graph_time.as_micros() as f64 / 100.0 / 1000.0
    );

    // Now evaluate all
    let start = Instant::now();
    for r in &results {
        eval([r])?;
    }
    let exec_time = start.elapsed();
    println!("Execution (100 iters): {:?}", exec_time);
    println!(
        "Per execution: {:.3} ms",
        exec_time.as_micros() as f64 / 100.0 / 1000.0
    );

    Ok(())
}

/// Benchmark generation loop timing with a real model.
///
/// This profiles each step of the generation loop to compare with mlx_lm's timing.
async fn run_gen_benchmark(model_id: &str) -> anyhow::Result<()> {
    use mlx_rs::{
        Array,
        ops::indexing::{IndexOp, argmax},
        transforms::{async_eval, eval},
    };
    use pmetal_models::DynamicModel;
    use std::path::PathBuf;
    use std::time::Instant;

    println!("=== Generation Loop Benchmark ===");
    println!("Model: {}\n", model_id);

    // Download model if needed
    let model_path = if model_id.contains('/') && !PathBuf::from(model_id).exists() {
        pmetal_hub::download_model(model_id, None, None).await?
    } else {
        PathBuf::from(model_id)
    };

    // Load model
    println!("Loading model...");
    let mut model = DynamicModel::load(&model_path)?;
    println!("Model loaded.\n");

    // Create KV cache
    let mut cache = model.create_cache(256);

    // Warmup
    println!("Warming up...");
    let token = Array::from_slice(&[42i32], &[1, 1]);
    let _ = model.forward_with_cache(&token, None, Some(&mut cache))?;
    eval([&token])?;
    cache = model.create_cache(256);
    println!("Warmup complete.\n");

    // Profile generation loop
    let n_tokens = 50;
    println!("Profiling {} token generation...\n", n_tokens);

    // Initial forward pass
    let logits = model.forward_with_cache(&token, None, Some(&mut cache))?;
    let mut current_token = argmax(&logits.index((.., -1, ..)), None)?;
    async_eval([&current_token])?;

    let mut times = std::collections::HashMap::new();
    times.insert("reshape", Vec::new());
    times.insert("forward", Vec::new());
    times.insert("extract_logits", Vec::new());
    times.insert("argmax", Vec::new());
    times.insert("async_eval", Vec::new());
    times.insert("item", Vec::new());
    times.insert("total", Vec::new());

    for _ in 0..n_tokens {
        let total_start = Instant::now();

        // Reshape token to [1, 1]
        let t0 = Instant::now();
        let next_input = current_token.reshape(&[1, 1])?;
        times.entry("reshape").or_default().push(t0.elapsed());

        // Forward pass
        let t0 = Instant::now();
        let next_logits = model.forward_with_cache(&next_input, None, Some(&mut cache))?;
        times.entry("forward").or_default().push(t0.elapsed());

        // Extract last logits
        let t0 = Instant::now();
        let last_logits = next_logits.index((.., -1, ..));
        times
            .entry("extract_logits")
            .or_default()
            .push(t0.elapsed());

        // Argmax
        let t0 = Instant::now();
        let next_token = argmax(&last_logits, None)?;
        times.entry("argmax").or_default().push(t0.elapsed());

        // Async eval for next
        let t0 = Instant::now();
        async_eval([&next_token])?;
        times.entry("async_eval").or_default().push(t0.elapsed());

        // Extract current token (sync point)
        let t0 = Instant::now();
        let _ = current_token.item::<u32>();
        times.entry("item").or_default().push(t0.elapsed());

        times
            .entry("total")
            .or_default()
            .push(total_start.elapsed());

        current_token = next_token;
    }

    // Print results
    println!("=== Generation Loop Timing ===");
    for (name, durations) in &times {
        let avg_us: f64 =
            durations.iter().map(|d| d.as_micros() as f64).sum::<f64>() / durations.len() as f64;
        let avg_ms = avg_us / 1000.0;
        println!("{:15}: {:7.3}ms", name, avg_ms);
    }

    let total_avg: f64 = times["total"]
        .iter()
        .map(|d| d.as_micros() as f64)
        .sum::<f64>()
        / times["total"].len() as f64;
    println!("\nEffective tok/s: {:.0}", 1_000_000.0 / total_avg);

    println!("\n=== Comparison ===");
    println!("Python mlx_lm reference:");
    println!("  build_graph: 0.570ms");
    println!("  async_eval:  3.111ms");
    println!("  item:        0.004ms");
    println!("  total:       3.686ms (271 tok/s)");

    Ok(())
}

/// Run dataset commands.
async fn run_dataset_command(action: DatasetAction) -> anyhow::Result<()> {
    use std::collections::HashMap;
    use std::io::{BufRead, BufReader, Write};

    match action {
        DatasetAction::Analyze {
            path,
            model,
            detailed,
        } => {
            println!("========================================");
            println!("  PMetal Dataset Analysis");
            println!("========================================");
            println!("Path: {}", path);
            println!("========================================\n");

            // Load tokenizer if model specified
            let tokenizer = if let Some(model_id) = &model {
                println!("Loading tokenizer from {}...", model_id);
                let model_path =
                    if model_id.contains('/') && !std::path::Path::new(model_id).exists() {
                        pmetal_hub::download_model(model_id, None, None).await?
                    } else {
                        std::path::PathBuf::from(model_id)
                    };
                Some(Tokenizer::from_model_dir(&model_path)?)
            } else {
                None
            };

            // Read JSONL file
            let file = std::fs::File::open(&path)?;
            let reader = BufReader::new(file);

            let mut total_samples = 0usize;
            let mut char_lengths = Vec::new();
            let mut token_lengths = Vec::new();
            let mut formats_detected: HashMap<String, usize> = HashMap::new();
            let mut empty_samples = 0usize;

            for line in reader.lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }

                total_samples += 1;

                // Parse JSON to detect format
                let json: serde_json::Value = serde_json::from_str(&line)?;

                // Detect format
                let format = if json.get("text").is_some() {
                    "simple"
                } else if json.get("conversations").is_some() {
                    "sharegpt"
                } else if json.get("instruction").is_some() {
                    "alpaca"
                } else if json.get("messages").is_some() {
                    "messages"
                } else {
                    "unknown"
                };
                *formats_detected.entry(format.to_string()).or_insert(0) += 1;

                // Extract text content
                let text = if let Some(t) = json.get("text").and_then(|v| v.as_str()) {
                    t.to_string()
                } else if let Some(convs) = json.get("conversations").and_then(|v| v.as_array()) {
                    convs
                        .iter()
                        .filter_map(|c| c.get("value").and_then(|v| v.as_str()))
                        .collect::<Vec<_>>()
                        .join(" ")
                } else if let Some(inst) = json.get("instruction").and_then(|v| v.as_str()) {
                    let input = json.get("input").and_then(|v| v.as_str()).unwrap_or("");
                    let output = json.get("output").and_then(|v| v.as_str()).unwrap_or("");
                    format!("{} {} {}", inst, input, output)
                } else {
                    String::new()
                };

                if text.is_empty() {
                    empty_samples += 1;
                    continue;
                }

                char_lengths.push(text.len());

                // Tokenize if tokenizer available
                if let Some(ref tok) = tokenizer {
                    let tokens = tok.encode(&text)?;
                    token_lengths.push(tokens.len());
                }
            }

            // Compute statistics
            println!("=== Dataset Statistics ===\n");
            println!("Total samples:    {}", total_samples);
            println!("Empty samples:    {}", empty_samples);
            println!("Valid samples:    {}", total_samples - empty_samples);
            println!();

            println!("Detected formats:");
            for (format, count) in &formats_detected {
                println!(
                    "  {}: {} ({:.1}%)",
                    format,
                    count,
                    100.0 * *count as f64 / total_samples as f64
                );
            }
            println!();

            if !char_lengths.is_empty() {
                char_lengths.sort();
                let min = char_lengths[0];
                let max = *char_lengths.last().unwrap();
                let mean = char_lengths.iter().sum::<usize>() as f64 / char_lengths.len() as f64;
                let median = char_lengths[char_lengths.len() / 2];
                let p90 = char_lengths[(char_lengths.len() as f64 * 0.9) as usize];
                let p95 = char_lengths[(char_lengths.len() as f64 * 0.95) as usize];

                println!("Character lengths:");
                println!("  Min:    {}", min);
                println!("  Max:    {}", max);
                println!("  Mean:   {:.0}", mean);
                println!("  Median: {}", median);
                println!("  P90:    {}", p90);
                println!("  P95:    {}", p95);
                println!();
            }

            if !token_lengths.is_empty() {
                token_lengths.sort();
                let min = token_lengths[0];
                let max = *token_lengths.last().unwrap();
                let mean = token_lengths.iter().sum::<usize>() as f64 / token_lengths.len() as f64;
                let median = token_lengths[token_lengths.len() / 2];
                let p90 = token_lengths[(token_lengths.len() as f64 * 0.9) as usize];
                let p95 = token_lengths[(token_lengths.len() as f64 * 0.95) as usize];
                let p99 = token_lengths[(token_lengths.len() as f64 * 0.99) as usize];

                println!("Token lengths:");
                println!("  Min:    {}", min);
                println!("  Max:    {}", max);
                println!("  Mean:   {:.0}", mean);
                println!("  Median: {}", median);
                println!("  P90:    {}", p90);
                println!("  P95:    {} (recommended max_seq_len)", p95);
                println!("  P99:    {}", p99);
                println!();

                // Histogram
                let buckets = [512, 1024, 2048, 4096, 8192, 16384, 32768];
                println!("Token length distribution:");
                for (i, &bucket) in buckets.iter().enumerate() {
                    let lower = if i == 0 { 0 } else { buckets[i - 1] };
                    let count = token_lengths
                        .iter()
                        .filter(|&&l| l > lower && l <= bucket)
                        .count();
                    let pct = 100.0 * count as f64 / token_lengths.len() as f64;
                    let bar = "█".repeat((pct / 2.0) as usize);
                    println!(
                        "  {:>5}-{:<5}: {:>5} ({:5.1}%) {}",
                        lower, bucket, count, pct, bar
                    );
                }
                let over_max = token_lengths
                    .iter()
                    .filter(|&&l| l > *buckets.last().unwrap())
                    .count();
                if over_max > 0 {
                    let pct = 100.0 * over_max as f64 / token_lengths.len() as f64;
                    println!(
                        "  >{:5}:     {:>5} ({:5.1}%)",
                        buckets.last().unwrap(),
                        over_max,
                        pct
                    );
                }
            }

            if detailed && !token_lengths.is_empty() {
                println!("\n=== Sample Details ===");
                for (i, len) in token_lengths.iter().take(10).enumerate() {
                    println!("  Sample {}: {} tokens", i + 1, len);
                }
                if token_lengths.len() > 10 {
                    println!("  ... and {} more samples", token_lengths.len() - 10);
                }
            }
        }

        DatasetAction::Download {
            dataset_id,
            split,
            output,
            revision,
        } => {
            println!("========================================");
            println!("  PMetal Dataset Download");
            println!("========================================");
            println!("Dataset:  {}", dataset_id);
            println!("Split:    {}", split);
            println!("========================================\n");

            // Download parquet files
            println!("Downloading dataset from HuggingFace Hub...");
            let parquet_paths = pmetal_hub::download_dataset_parquet(
                &dataset_id,
                &split,
                revision.as_deref(),
                None,
            )
            .await?;

            println!("Downloaded {} parquet file(s)", parquet_paths.len());

            // Determine output path
            let output_path = output.unwrap_or_else(|| {
                let safe_name = dataset_id.replace('/', "_");
                format!("{}.jsonl", safe_name)
            });

            let validated_download_output =
                validate_output_path(&output_path, "dataset download output")?;
            println!(
                "Converting to JSONL: {}",
                validated_download_output.display()
            );

            // Convert parquet to JSONL using arrow-parquet
            let mut output_file = std::fs::File::create(&validated_download_output)?;
            let mut total_rows = 0usize;

            for parquet_path in &parquet_paths {
                use arrow_array::RecordBatchReader;
                use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

                let file = std::fs::File::open(parquet_path)?;
                let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
                let reader = builder.build()?;

                // Get column names from schema
                let schema = reader.schema();
                let columns: Vec<String> =
                    schema.fields().iter().map(|f| f.name().clone()).collect();
                println!("  Columns: {:?}", columns);

                // Use arrow-json to properly serialize record batches
                use arrow_json::writer::LineDelimitedWriter;

                for batch_result in reader {
                    let batch = batch_result?;

                    // Use arrow-json for proper nested type serialization
                    let mut json_buf = Vec::new();
                    {
                        let mut json_writer = LineDelimitedWriter::new(&mut json_buf);
                        json_writer.write(&batch)?;
                        json_writer.finish()?;
                    }

                    // Parse and re-write each line to handle conversions
                    for line in std::str::from_utf8(&json_buf)?.lines() {
                        if line.trim().is_empty() {
                            continue;
                        }

                        // Parse and potentially transform the JSON
                        let mut obj: serde_json::Value = serde_json::from_str(line)?;

                        // If it has "conversations" field as a string, parse it
                        if let Some(serde_json::Value::String(conv_str)) = obj.get("conversations")
                        {
                            if let Ok(convs) = serde_json::from_str::<serde_json::Value>(conv_str) {
                                obj["conversations"] = convs;
                            }
                        }

                        // If it has "messages" field, convert to ShareGPT format
                        if let Some(serde_json::Value::Array(msgs)) = obj.get("messages").cloned() {
                            let conversations: Vec<_> = msgs
                                .iter()
                                .filter_map(|m| {
                                    let role = m.get("role")?.as_str()?;
                                    let content = m.get("content")?.as_str()?;
                                    let from = match role {
                                        "user" => "human",
                                        "assistant" => "gpt",
                                        "system" => "system",
                                        _ => role,
                                    };
                                    Some(serde_json::json!({
                                        "from": from,
                                        "value": content
                                    }))
                                })
                                .collect();

                            // Replace messages with conversations in ShareGPT format
                            if let Some(obj_map) = obj.as_object_mut() {
                                obj_map.remove("messages");
                                obj_map.insert(
                                    "conversations".to_string(),
                                    serde_json::Value::Array(conversations),
                                );
                            }
                        }

                        writeln!(output_file, "{}", serde_json::to_string(&obj)?)?;
                        total_rows += 1;
                    }
                }
            }

            println!("\n========================================");
            println!("  Download Complete!");
            println!("========================================");
            println!("Total samples: {}", total_rows);
            println!("Output:        {}", output_path);
            println!("========================================");
        }

        DatasetAction::Convert {
            input,
            output,
            format,
            columns,
            shuffle,
            seed,
        } => {
            println!("========================================");
            println!("  PMetal Dataset Conversion");
            println!("========================================");
            println!("Input:    {}", input);
            println!("Output:   {}", output);
            if let Some(ref f) = format {
                println!("Format:   {:?}", f);
            }
            println!("Shuffle:  {}", shuffle);
            println!("========================================\n");

            // Parse column mappings
            let col_map: HashMap<String, String> = columns
                .map(|c| {
                    c.split(',')
                        .filter_map(|pair| {
                            let parts: Vec<&str> = pair.split('=').collect();
                            if parts.len() == 2 {
                                Some((parts[0].to_string(), parts[1].to_string()))
                            } else {
                                None
                            }
                        })
                        .collect()
                })
                .unwrap_or_default();

            // Read input
            let mut samples: Vec<serde_json::Value> = Vec::new();

            let input_path = std::path::Path::new(&input);
            let extension = input_path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("");

            match extension.to_lowercase().as_str() {
                "parquet" => {
                    use arrow_array::RecordBatchReader;
                    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

                    let file = std::fs::File::open(&input)?;
                    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
                    let reader = builder.build()?;

                    let schema = reader.schema();
                    let columns: Vec<String> =
                        schema.fields().iter().map(|f| f.name().clone()).collect();

                    for batch_result in reader {
                        let batch = batch_result?;
                        let num_rows = batch.num_rows();

                        for row_idx in 0..num_rows {
                            let mut obj = serde_json::Map::new();

                            for (col_idx, col_name) in columns.iter().enumerate() {
                                let target_name = col_map.get(col_name).unwrap_or(col_name);
                                let col = batch.column(col_idx);

                                use arrow_array::{Array, cast::AsArray};
                                let value = if let Some(arr) = col.as_string_opt::<i32>() {
                                    if arr.is_null(row_idx) {
                                        serde_json::Value::Null
                                    } else {
                                        serde_json::Value::String(arr.value(row_idx).to_string())
                                    }
                                } else if let Some(arr) = col.as_string_opt::<i64>() {
                                    if arr.is_null(row_idx) {
                                        serde_json::Value::Null
                                    } else {
                                        serde_json::Value::String(arr.value(row_idx).to_string())
                                    }
                                } else {
                                    serde_json::Value::Null
                                };

                                obj.insert(target_name.clone(), value);
                            }
                            samples.push(serde_json::Value::Object(obj));
                        }
                    }
                }
                "jsonl" | "json" => {
                    let file = std::fs::File::open(&input)?;
                    let reader = BufReader::new(file);

                    for line in reader.lines() {
                        let line = line?;
                        if line.trim().is_empty() {
                            continue;
                        }
                        let obj: serde_json::Value = serde_json::from_str(&line)?;
                        samples.push(obj);
                    }
                }
                _ => {
                    return Err(anyhow::anyhow!("Unsupported input format: {}", extension));
                }
            }

            println!("Loaded {} samples", samples.len());

            // Shuffle if requested
            if shuffle {
                use rand::SeedableRng;
                use rand::seq::SliceRandom;
                let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
                samples.shuffle(&mut rng);
                println!("Shuffled with seed {}", seed);
            }

            // Validate and write output
            let validated_convert_output = validate_output_path(&output, "dataset convert output")?;
            let mut output_file = std::fs::File::create(&validated_convert_output)?;
            for sample in &samples {
                writeln!(output_file, "{}", serde_json::to_string(sample)?)?;
            }

            println!("\n========================================");
            println!("  Conversion Complete!");
            println!("========================================");
            println!("Output:  {}", output);
            println!("Samples: {}", samples.len());
            println!("========================================");
        }

        DatasetAction::Validate {
            path,
            model,
            max_seq_len,
        } => {
            println!("========================================");
            println!("  PMetal Dataset Validation");
            println!("========================================");
            println!("Path:        {}", path);
            println!("Max Seq Len: {}", max_seq_len);
            println!("========================================\n");

            // Load tokenizer
            let tokenizer = if let Some(model_id) = &model {
                println!("Loading tokenizer from {}...", model_id);
                let model_path =
                    if model_id.contains('/') && !std::path::Path::new(model_id).exists() {
                        pmetal_hub::download_model(model_id, None, None).await?
                    } else {
                        std::path::PathBuf::from(model_id)
                    };
                Some(Tokenizer::from_model_dir(&model_path)?)
            } else {
                None
            };

            // Read JSONL file
            let file = std::fs::File::open(&path)?;
            let reader = BufReader::new(file);

            let mut total_samples = 0usize;
            let mut valid_samples = 0usize;
            let mut too_long = 0usize;
            let mut empty = 0usize;
            let mut parse_errors = 0usize;
            let mut issues: Vec<String> = Vec::new();

            for (line_num, line) in reader.lines().enumerate() {
                let line = match line {
                    Ok(l) => l,
                    Err(e) => {
                        issues.push(format!("Line {}: Read error: {}", line_num + 1, e));
                        parse_errors += 1;
                        continue;
                    }
                };

                if line.trim().is_empty() {
                    continue;
                }

                total_samples += 1;

                // Parse JSON
                let json: serde_json::Value = match serde_json::from_str(&line) {
                    Ok(j) => j,
                    Err(e) => {
                        issues.push(format!("Line {}: JSON parse error: {}", line_num + 1, e));
                        parse_errors += 1;
                        continue;
                    }
                };

                // Extract text
                let text = if let Some(t) = json.get("text").and_then(|v| v.as_str()) {
                    t.to_string()
                } else if let Some(convs) = json.get("conversations").and_then(|v| v.as_array()) {
                    convs
                        .iter()
                        .filter_map(|c| c.get("value").and_then(|v| v.as_str()))
                        .collect::<Vec<_>>()
                        .join(" ")
                } else if let Some(inst) = json.get("instruction").and_then(|v| v.as_str()) {
                    let input = json.get("input").and_then(|v| v.as_str()).unwrap_or("");
                    let output = json.get("output").and_then(|v| v.as_str()).unwrap_or("");
                    format!("{} {} {}", inst, input, output)
                } else {
                    issues.push(format!("Line {}: No recognizable text field", line_num + 1));
                    empty += 1;
                    continue;
                };

                if text.trim().is_empty() {
                    issues.push(format!("Line {}: Empty text content", line_num + 1));
                    empty += 1;
                    continue;
                }

                // Tokenize and check length
                if let Some(ref tok) = tokenizer {
                    let tokens = tok.encode(&text)?;
                    if tokens.len() > max_seq_len {
                        too_long += 1;
                        if issues.len() < 10 {
                            issues.push(format!(
                                "Line {}: Token length {} exceeds max_seq_len {}",
                                line_num + 1,
                                tokens.len(),
                                max_seq_len
                            ));
                        }
                        continue;
                    }
                }

                valid_samples += 1;
            }

            // Report results
            println!("=== Validation Results ===\n");
            println!("Total samples:   {}", total_samples);
            println!(
                "Valid samples:   {} ({:.1}%)",
                valid_samples,
                100.0 * valid_samples as f64 / total_samples as f64
            );
            println!("Parse errors:    {}", parse_errors);
            println!("Empty samples:   {}", empty);
            println!("Too long:        {} (>{} tokens)", too_long, max_seq_len);
            println!();

            if !issues.is_empty() {
                println!("Issues (first 10):");
                for issue in issues.iter().take(10) {
                    println!("  - {}", issue);
                }
                if issues.len() > 10 {
                    println!("  ... and {} more issues", issues.len() - 10);
                }
            }

            // Overall status
            if parse_errors == 0 && empty == 0 {
                println!("\n✓ Dataset is valid for training");
                if too_long > 0 {
                    println!(
                        "  Note: {} samples exceed max_seq_len and will be truncated",
                        too_long
                    );
                }
            } else {
                println!("\n✗ Dataset has issues that need attention");
            }
        }

        DatasetAction::Preview {
            dataset_id,
            split,
            num,
        } => {
            println!("========================================");
            println!("  PMetal Dataset Preview");
            println!("========================================");
            println!("Dataset: {}", dataset_id);
            println!("Split:   {}", split);
            println!("Samples: {}", num);
            println!("========================================\n");

            // Download parquet files
            println!("Fetching dataset...");
            let parquet_paths =
                pmetal_hub::download_dataset_parquet(&dataset_id, &split, None, None).await?;

            let mut shown = 0usize;
            'outer: for parquet_path in &parquet_paths {
                use arrow_array::RecordBatchReader;
                use arrow_json::writer::LineDelimitedWriter;
                use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

                let file = std::fs::File::open(parquet_path)?;
                let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
                let reader = builder.build()?;

                let schema = reader.schema();
                if shown == 0 {
                    let columns: Vec<&str> =
                        schema.fields().iter().map(|f| f.name().as_str()).collect();
                    println!("Columns: {:?}\n", columns);
                }

                for batch_result in reader {
                    let batch = batch_result?;

                    // Serialize batch to JSON
                    let mut json_buf = Vec::new();
                    {
                        let mut json_writer = LineDelimitedWriter::new(&mut json_buf);
                        json_writer.write(&batch)?;
                        json_writer.finish()?;
                    }

                    for line in std::str::from_utf8(&json_buf)?.lines() {
                        if line.trim().is_empty() {
                            continue;
                        }
                        if shown >= num {
                            break 'outer;
                        }

                        let obj: serde_json::Value = serde_json::from_str(line)?;
                        println!("--- Sample {} ---", shown + 1);
                        println!("{}", serde_json::to_string_pretty(&obj)?);
                        println!();
                        shown += 1;
                    }
                }
            }

            println!("Showed {} sample(s)", shown);
        }

        DatasetAction::Filter {
            input,
            output,
            model,
            min_tokens,
            max_tokens,
            dedup,
            pattern,
            invert,
            complete_only,
        } => {
            println!("========================================");
            println!("  PMetal Dataset Filter");
            println!("========================================");
            println!("Input:  {}", input);
            println!("Output: {}", output);
            if let Some(min) = min_tokens {
                println!("Min tokens: {}", min);
            }
            if let Some(max) = max_tokens {
                println!("Max tokens: {}", max);
            }
            if dedup {
                println!("Deduplication: enabled");
            }
            if let Some(ref p) = pattern {
                println!("Pattern: {} (invert: {})", p, invert);
            }
            if complete_only {
                println!("Complete only: enabled");
            }
            println!("========================================\n");

            // Load tokenizer if needed for token filtering
            let tokenizer = if min_tokens.is_some() || max_tokens.is_some() {
                if let Some(model_id) = &model {
                    println!("Loading tokenizer from {}...", model_id);
                    let model_path =
                        if model_id.contains('/') && !std::path::Path::new(model_id).exists() {
                            pmetal_hub::download_model(model_id, None, None).await?
                        } else {
                            std::path::PathBuf::from(model_id)
                        };
                    Some(Tokenizer::from_model_dir(&model_path)?)
                } else {
                    return Err(anyhow::anyhow!(
                        "--model required for token-based filtering"
                    ));
                }
            } else {
                None
            };

            // Compile regex if provided
            let regex = if let Some(ref p) = pattern {
                Some(regex::Regex::new(p)?)
            } else {
                None
            };

            // For deduplication
            let mut seen_hashes: std::collections::HashSet<u64> = std::collections::HashSet::new();

            let validated_filter_output = validate_output_path(&output, "dataset filter output")?;
            let file = std::fs::File::open(&input)?;
            let reader = BufReader::new(file);
            let mut output_file = std::fs::File::create(&validated_filter_output)?;

            let mut total = 0usize;
            let mut kept = 0usize;
            let mut filtered_tokens = 0usize;
            let mut filtered_pattern = 0usize;
            let mut filtered_dedup = 0usize;
            let mut filtered_incomplete = 0usize;

            for line in reader.lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                total += 1;

                let json: serde_json::Value = serde_json::from_str(&line)?;

                // Extract text for filtering
                let text = extract_text_from_sample(&json);

                // Check completeness for ShareGPT
                if complete_only {
                    if let Some(convs) = json.get("conversations").and_then(|v| v.as_array()) {
                        let has_human = convs
                            .iter()
                            .any(|c| c.get("from").and_then(|v| v.as_str()) == Some("human"));
                        let has_gpt = convs
                            .iter()
                            .any(|c| c.get("from").and_then(|v| v.as_str()) == Some("gpt"));
                        if !has_human || !has_gpt {
                            filtered_incomplete += 1;
                            continue;
                        }
                    }
                }

                // Token length filtering
                if let Some(ref tok) = tokenizer {
                    let tokens = tok.encode(&text)?;
                    let len = tokens.len();
                    if let Some(min) = min_tokens {
                        if len < min {
                            filtered_tokens += 1;
                            continue;
                        }
                    }
                    if let Some(max) = max_tokens {
                        if len > max {
                            filtered_tokens += 1;
                            continue;
                        }
                    }
                }

                // Pattern filtering
                if let Some(ref re) = regex {
                    let matches = re.is_match(&text);
                    let keep = if invert { !matches } else { matches };
                    if !keep {
                        filtered_pattern += 1;
                        continue;
                    }
                }

                // Deduplication
                if dedup {
                    use std::hash::{Hash, Hasher};
                    let mut hasher = std::collections::hash_map::DefaultHasher::new();
                    text.hash(&mut hasher);
                    let hash = hasher.finish();
                    if !seen_hashes.insert(hash) {
                        filtered_dedup += 1;
                        continue;
                    }
                }

                // Keep this sample
                writeln!(output_file, "{}", line)?;
                kept += 1;
            }

            println!("\n========================================");
            println!("  Filter Complete!");
            println!("========================================");
            println!("Total samples:        {}", total);
            println!(
                "Kept samples:         {} ({:.1}%)",
                kept,
                100.0 * kept as f64 / total as f64
            );
            if filtered_tokens > 0 {
                println!("Filtered (tokens):    {}", filtered_tokens);
            }
            if filtered_pattern > 0 {
                println!("Filtered (pattern):   {}", filtered_pattern);
            }
            if filtered_dedup > 0 {
                println!("Filtered (duplicate): {}", filtered_dedup);
            }
            if filtered_incomplete > 0 {
                println!("Filtered (incomplete):{}", filtered_incomplete);
            }
            println!("Output: {}", output);
            println!("========================================");
        }

        DatasetAction::Split {
            input,
            output_dir,
            val_ratio,
            test_ratio,
            seed,
            stratify,
        } => {
            println!("========================================");
            println!("  PMetal Dataset Split");
            println!("========================================");
            println!("Input:     {}", input);
            println!("Output:    {}", output_dir);
            println!("Val ratio: {:.2}", val_ratio);
            println!("Test ratio:{:.2}", test_ratio);
            println!("Seed:      {}", seed);
            if let Some(ref s) = stratify {
                println!("Stratify:  {}", s);
            }
            println!("========================================\n");

            // Validate ratios
            if val_ratio + test_ratio >= 1.0 {
                return Err(anyhow::anyhow!("val_ratio + test_ratio must be < 1.0"));
            }

            // Read all samples
            let file = std::fs::File::open(&input)?;
            let reader = BufReader::new(file);
            let mut samples: Vec<String> = Vec::new();

            for line in reader.lines() {
                let line = line?;
                if !line.trim().is_empty() {
                    samples.push(line);
                }
            }

            println!("Loaded {} samples", samples.len());

            // Shuffle
            use rand::SeedableRng;
            use rand::seq::SliceRandom;
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            samples.shuffle(&mut rng);

            // Calculate split indices
            let total = samples.len();
            let test_count = (total as f64 * test_ratio).round() as usize;
            let val_count = (total as f64 * val_ratio).round() as usize;
            let train_count = total - test_count - val_count;

            // Validate and create output directory
            let validated_split_dir = validate_output_path(&output_dir, "dataset split output")?;
            std::fs::create_dir_all(&validated_split_dir)?;

            // Write splits
            let train_path = validated_split_dir.join("train.jsonl");
            let val_path = validated_split_dir.join("val.jsonl");
            let test_path = validated_split_dir.join("test.jsonl");

            let mut train_file = std::fs::File::create(&train_path)?;
            for sample in samples.iter().take(train_count) {
                writeln!(train_file, "{}", sample)?;
            }
            println!("Train: {} samples -> {}", train_count, train_path.display());

            let mut val_file = std::fs::File::create(&val_path)?;
            for sample in samples.iter().skip(train_count).take(val_count) {
                writeln!(val_file, "{}", sample)?;
            }
            println!("Val:   {} samples -> {}", val_count, val_path.display());

            if test_count > 0 {
                let mut test_file = std::fs::File::create(&test_path)?;
                for sample in samples.iter().skip(train_count + val_count) {
                    writeln!(test_file, "{}", sample)?;
                }
                println!("Test:  {} samples -> {}", test_count, test_path.display());
            }

            println!("\n========================================");
            println!("  Split Complete!");
            println!("========================================");
        }

        DatasetAction::Merge {
            inputs,
            output,
            shuffle,
            seed,
            interleave,
            weights,
        } => {
            println!("========================================");
            println!("  PMetal Dataset Merge");
            println!("========================================");
            for (i, input) in inputs.iter().enumerate() {
                println!("Input {}: {}", i + 1, input);
            }
            println!("Output:     {}", output);
            println!("Shuffle:    {}", shuffle);
            println!("Interleave: {}", interleave);
            println!("========================================\n");

            // Parse weights if provided
            let weights_vec: Vec<f64> = if let Some(ref w) = weights {
                w.split(',')
                    .map(|s| s.trim().parse::<f64>().unwrap_or(1.0))
                    .collect()
            } else {
                vec![1.0; inputs.len()]
            };

            // Read all datasets
            let mut all_samples: Vec<Vec<String>> = Vec::new();
            for input in &inputs {
                let file = std::fs::File::open(input)?;
                let reader = BufReader::new(file);
                let samples: Vec<String> = reader
                    .lines()
                    .map_while(Result::ok)
                    .filter(|l| !l.trim().is_empty())
                    .collect();
                println!("Loaded {} samples from {}", samples.len(), input);
                all_samples.push(samples);
            }

            let mut merged: Vec<String> = Vec::new();

            if interleave {
                // Interleave samples from each dataset
                let max_len = all_samples.iter().map(|s| s.len()).max().unwrap_or(0);
                for i in 0..max_len {
                    for (dataset_idx, samples) in all_samples.iter().enumerate() {
                        let weight = weights_vec.get(dataset_idx).copied().unwrap_or(1.0);
                        if i < samples.len() && rand::random::<f64>() < weight {
                            merged.push(samples[i].clone());
                        }
                    }
                }
            } else {
                // Simple concatenation with optional weighting (sampling)
                for (dataset_idx, samples) in all_samples.iter().enumerate() {
                    let weight = weights_vec.get(dataset_idx).copied().unwrap_or(1.0);
                    if weight >= 1.0 {
                        // Include all samples, possibly multiple times
                        let repeat = weight.floor() as usize;
                        for _ in 0..repeat.max(1) {
                            merged.extend(samples.iter().cloned());
                        }
                    } else {
                        // Sample a fraction
                        use rand::SeedableRng;
                        let mut rng = rand::rngs::StdRng::seed_from_u64(seed + dataset_idx as u64);
                        for sample in samples {
                            if rand::RngExt::random::<f64>(&mut rng) < weight {
                                merged.push(sample.clone());
                            }
                        }
                    }
                }
            }

            // Shuffle if requested
            if shuffle {
                use rand::SeedableRng;
                use rand::seq::SliceRandom;
                let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
                merged.shuffle(&mut rng);
                println!("Shuffled with seed {}", seed);
            }

            // Write output
            let mut output_file = std::fs::File::create(&output)?;
            for sample in &merged {
                writeln!(output_file, "{}", sample)?;
            }

            println!("\n========================================");
            println!("  Merge Complete!");
            println!("========================================");
            println!("Total samples: {}", merged.len());
            println!("Output: {}", output);
            println!("========================================");
        }

        DatasetAction::Sample {
            input,
            output,
            num,
            seed,
        } => {
            println!("========================================");
            println!("  PMetal Dataset Sample");
            println!("========================================");
            println!("Input:   {}", input);
            println!("Output:  {}", output);
            println!("Samples: {}", num);
            println!("Seed:    {}", seed);
            println!("========================================\n");

            // Read all samples
            let file = std::fs::File::open(&input)?;
            let reader = BufReader::new(file);
            let mut samples: Vec<String> = reader
                .lines()
                .map_while(Result::ok)
                .filter(|l| !l.trim().is_empty())
                .collect();

            println!("Loaded {} samples", samples.len());

            // Shuffle and take first N
            use rand::SeedableRng;
            use rand::seq::SliceRandom;
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            samples.shuffle(&mut rng);

            let take_count = num.min(samples.len());
            let mut output_file = std::fs::File::create(&output)?;
            for sample in samples.iter().take(take_count) {
                writeln!(output_file, "{}", sample)?;
            }

            println!("\n========================================");
            println!("  Sample Complete!");
            println!("========================================");
            println!("Sampled {} of {} samples", take_count, samples.len());
            println!("Output: {}", output);
            println!("========================================");
        }

        DatasetAction::Template {
            input,
            output,
            template,
            system,
            model: _,
            add_generation_prompt,
            mask_prompt: _,
        } => {
            println!("========================================");
            println!("  PMetal Chat Template");
            println!("========================================");
            println!("Input:    {}", input);
            println!("Output:   {}", output);
            println!("Template: {:?}", template);
            if let Some(ref s) = system {
                println!("System:   {}", s);
            }
            if add_generation_prompt {
                println!("Add generation prompt: yes");
            }
            println!("========================================\n");

            let file = std::fs::File::open(&input)?;
            let reader = BufReader::new(file);
            let mut output_file = std::fs::File::create(&output)?;

            let mut total = 0usize;
            let mut templated = 0usize;

            for line in reader.lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                total += 1;

                let json: serde_json::Value = serde_json::from_str(&line)?;

                // Extract conversations
                let conversations =
                    if let Some(convs) = json.get("conversations").and_then(|v| v.as_array()) {
                        convs
                            .iter()
                            .filter_map(|c| {
                                let from = c.get("from")?.as_str()?;
                                let value = c.get("value")?.as_str()?;
                                Some((from.to_string(), value.to_string()))
                            })
                            .collect::<Vec<_>>()
                    } else if let Some(inst) = json.get("instruction").and_then(|v| v.as_str()) {
                        // Convert Alpaca to conversations
                        let input_text = json.get("input").and_then(|v| v.as_str()).unwrap_or("");
                        let output_text = json.get("output").and_then(|v| v.as_str()).unwrap_or("");
                        let user_msg = if input_text.is_empty() {
                            inst.to_string()
                        } else {
                            format!("{}\n\n{}", inst, input_text)
                        };
                        vec![
                            ("human".to_string(), user_msg),
                            ("gpt".to_string(), output_text.to_string()),
                        ]
                    } else if let Some(text) = json.get("text").and_then(|v| v.as_str()) {
                        // Already raw text, just wrap it
                        writeln!(output_file, "{}", serde_json::json!({"text": text}))?;
                        templated += 1;
                        continue;
                    } else {
                        // Skip samples without recognizable format
                        continue;
                    };

                // Apply chat template
                let formatted = format_conversations(
                    &template,
                    &conversations,
                    system.as_deref(),
                    add_generation_prompt,
                );

                // Write output
                let out_json = serde_json::json!({ "text": formatted });
                writeln!(output_file, "{}", out_json)?;
                templated += 1;
            }

            println!("\n========================================");
            println!("  Template Complete!");
            println!("========================================");
            println!("Total samples:     {}", total);
            println!("Templated samples: {}", templated);
            println!("Output: {}", output);
            println!("========================================");
        }

        DatasetAction::Prepare {
            dataset,
            output_dir,
            model,
            template,
            max_seq_len,
            val_ratio,
            seed,
            no_dedup,
        } => {
            println!("========================================");
            println!("  PMetal Dataset Prepare");
            println!("========================================");
            println!("Dataset:     {}", dataset);
            println!("Output:      {}", output_dir);
            println!("Model:       {}", model);
            println!("Template:    {:?}", template);
            println!("Max seq len: {}", max_seq_len);
            println!("Val ratio:   {:.2}", val_ratio);
            println!("Seed:        {}", seed);
            println!("========================================\n");

            // Create output directory
            std::fs::create_dir_all(&output_dir)?;

            // Step 1: Download or load dataset
            println!("[1/5] Loading dataset...");
            let raw_path = format!("{}/raw.jsonl", output_dir);

            if dataset.contains('/') && !std::path::Path::new(&dataset).exists() {
                // HuggingFace dataset
                let parquet_paths =
                    pmetal_hub::download_dataset_parquet(&dataset, "train", None, None).await?;

                // Convert to JSONL
                let mut output_file = std::fs::File::create(&raw_path)?;
                let mut total_rows = 0usize;

                for parquet_path in &parquet_paths {
                    use arrow_json::writer::LineDelimitedWriter;
                    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

                    let file = std::fs::File::open(parquet_path)?;
                    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
                    let reader = builder.build()?;

                    for batch_result in reader {
                        let batch = batch_result?;

                        let mut json_buf = Vec::new();
                        {
                            let mut json_writer = LineDelimitedWriter::new(&mut json_buf);
                            json_writer.write(&batch)?;
                            json_writer.finish()?;
                        }

                        for line in std::str::from_utf8(&json_buf)?.lines() {
                            if line.trim().is_empty() {
                                continue;
                            }

                            let mut obj: serde_json::Value = serde_json::from_str(line)?;

                            // Convert messages to ShareGPT format if needed
                            if let Some(serde_json::Value::Array(msgs)) =
                                obj.get("messages").cloned()
                            {
                                let conversations: Vec<_> = msgs
                                    .iter()
                                    .filter_map(|m| {
                                        let role = m.get("role")?.as_str()?;
                                        let content = m.get("content")?.as_str()?;
                                        let from = match role {
                                            "user" => "human",
                                            "assistant" => "gpt",
                                            "system" => "system",
                                            _ => role,
                                        };
                                        Some(serde_json::json!({
                                            "from": from,
                                            "value": content
                                        }))
                                    })
                                    .collect();

                                if let Some(obj_map) = obj.as_object_mut() {
                                    obj_map.remove("messages");
                                    obj_map.insert(
                                        "conversations".to_string(),
                                        serde_json::Value::Array(conversations),
                                    );
                                }
                            }

                            writeln!(output_file, "{}", serde_json::to_string(&obj)?)?;
                            total_rows += 1;
                        }
                    }
                }
                println!("  Downloaded {} samples", total_rows);
            } else {
                // Local file - copy
                std::fs::copy(&dataset, &raw_path)?;
                println!("  Copied local dataset");
            }

            // Step 2: Load tokenizer
            println!("[2/5] Loading tokenizer...");
            let model_path = if model.contains('/') && !std::path::Path::new(&model).exists() {
                pmetal_hub::download_model(&model, None, None).await?
            } else {
                std::path::PathBuf::from(&model)
            };
            let tokenizer = Tokenizer::from_model_dir(&model_path)?;
            println!("  Loaded tokenizer from {}", model_path.display());

            // Step 3: Apply template and filter
            println!("[3/5] Applying template and filtering...");
            let templated_path = format!("{}/templated.jsonl", output_dir);
            let raw_file = std::fs::File::open(&raw_path)?;
            let raw_reader = BufReader::new(raw_file);
            let mut templated_file = std::fs::File::create(&templated_path)?;

            let mut seen_hashes: std::collections::HashSet<u64> = std::collections::HashSet::new();
            let mut total = 0usize;
            let mut kept = 0usize;
            let mut filtered_long = 0usize;
            let mut filtered_dedup = 0usize;

            for line in raw_reader.lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                total += 1;

                let json: serde_json::Value = serde_json::from_str(&line)?;

                // Extract conversations
                let conversations =
                    if let Some(convs) = json.get("conversations").and_then(|v| v.as_array()) {
                        convs
                            .iter()
                            .filter_map(|c| {
                                let from = c.get("from")?.as_str()?;
                                let value = c.get("value")?.as_str()?;
                                Some((from.to_string(), value.to_string()))
                            })
                            .collect::<Vec<_>>()
                    } else if let Some(inst) = json.get("instruction").and_then(|v| v.as_str()) {
                        let input_text = json.get("input").and_then(|v| v.as_str()).unwrap_or("");
                        let output_text = json.get("output").and_then(|v| v.as_str()).unwrap_or("");
                        let user_msg = if input_text.is_empty() {
                            inst.to_string()
                        } else {
                            format!("{}\n\n{}", inst, input_text)
                        };
                        vec![
                            ("human".to_string(), user_msg),
                            ("gpt".to_string(), output_text.to_string()),
                        ]
                    } else if let Some(text) = json.get("text").and_then(|v| v.as_str()) {
                        // Check length
                        let tokens = tokenizer.encode(text)?;
                        if tokens.len() > max_seq_len {
                            filtered_long += 1;
                            continue;
                        }

                        // Check dedup
                        if !no_dedup {
                            use std::hash::{Hash, Hasher};
                            let mut hasher = std::collections::hash_map::DefaultHasher::new();
                            text.hash(&mut hasher);
                            let hash = hasher.finish();
                            if !seen_hashes.insert(hash) {
                                filtered_dedup += 1;
                                continue;
                            }
                        }

                        writeln!(templated_file, "{}", serde_json::json!({"text": text}))?;
                        kept += 1;
                        continue;
                    } else {
                        continue;
                    };

                // Apply template
                let formatted = format_conversations(&template, &conversations, None, false);

                // Check token length
                let tokens = tokenizer.encode(&formatted)?;
                if tokens.len() > max_seq_len {
                    filtered_long += 1;
                    continue;
                }

                // Check dedup
                if !no_dedup {
                    use std::hash::{Hash, Hasher};
                    let mut hasher = std::collections::hash_map::DefaultHasher::new();
                    formatted.hash(&mut hasher);
                    let hash = hasher.finish();
                    if !seen_hashes.insert(hash) {
                        filtered_dedup += 1;
                        continue;
                    }
                }

                writeln!(templated_file, "{}", serde_json::json!({"text": formatted}))?;
                kept += 1;
            }

            println!(
                "  Total: {}, Kept: {}, Filtered (long): {}, Filtered (dup): {}",
                total, kept, filtered_long, filtered_dedup
            );

            // Step 4: Split
            println!("[4/5] Splitting dataset...");
            let templated_file = std::fs::File::open(&templated_path)?;
            let templated_reader = BufReader::new(templated_file);
            let mut samples: Vec<String> = templated_reader
                .lines()
                .map_while(Result::ok)
                .filter(|l| !l.trim().is_empty())
                .collect();

            use rand::SeedableRng;
            use rand::seq::SliceRandom;
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            samples.shuffle(&mut rng);

            let val_count = (samples.len() as f64 * val_ratio).round() as usize;
            let train_count = samples.len() - val_count;

            let train_path = format!("{}/train.jsonl", output_dir);
            let val_path = format!("{}/val.jsonl", output_dir);

            let mut train_file = std::fs::File::create(&train_path)?;
            for sample in samples.iter().take(train_count) {
                writeln!(train_file, "{}", sample)?;
            }

            let mut val_file = std::fs::File::create(&val_path)?;
            for sample in samples.iter().skip(train_count) {
                writeln!(val_file, "{}", sample)?;
            }

            println!("  Train: {} samples", train_count);
            println!("  Val:   {} samples", val_count);

            // Step 5: Statistics
            println!("[5/5] Computing statistics...");
            let train_file = std::fs::File::open(&train_path)?;
            let train_reader = BufReader::new(train_file);
            let mut token_lengths: Vec<usize> = Vec::new();

            for line in train_reader.lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                let json: serde_json::Value = serde_json::from_str(&line)?;
                if let Some(text) = json.get("text").and_then(|v| v.as_str()) {
                    let tokens = tokenizer.encode(text)?;
                    token_lengths.push(tokens.len());
                }
            }

            token_lengths.sort();
            let p50 = token_lengths[token_lengths.len() / 2];
            let p95 = token_lengths[(token_lengths.len() as f64 * 0.95) as usize];
            let max_len = *token_lengths.last().unwrap_or(&0);

            println!("\n========================================");
            println!("  Prepare Complete!");
            println!("========================================");
            println!("Train samples: {}", train_count);
            println!("Val samples:   {}", val_count);
            println!("Token P50:     {}", p50);
            println!("Token P95:     {}", p95);
            println!("Token Max:     {}", max_len);
            println!("\nOutput files:");
            println!("  {}", train_path);
            println!("  {}", val_path);
            println!("========================================");
        }

        DatasetAction::Formats => {
            println!("========================================");
            println!("  PMetal Supported Formats & Templates");
            println!("========================================\n");

            println!("INPUT FORMATS:");
            println!("--------------");
            println!("1. ShareGPT (recommended):");
            println!(
                r#"   {{"conversations": [{{"from": "human", "value": "..."}}, {{"from": "gpt", "value": "..."}}]}}"#
            );
            println!();
            println!("2. Alpaca:");
            println!(r#"   {{"instruction": "...", "input": "...", "output": "..."}}"#);
            println!();
            println!("3. OpenAI Messages:");
            println!(
                r#"   {{"messages": [{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}]}}"#
            );
            println!();
            println!("4. Simple text:");
            println!(r#"   {{"text": "The full formatted text for training"}}"#);
            println!();

            println!("CHAT TEMPLATES:");
            println!("---------------");
            println!("1. chatml (default):");
            println!("   <|im_start|>system");
            println!("   {{system_message}}<|im_end|>");
            println!("   <|im_start|>user");
            println!("   {{user_message}}<|im_end|>");
            println!("   <|im_start|>assistant");
            println!("   {{assistant_message}}<|im_end|>");
            println!();
            println!("2. llama3:");
            println!("   <|start_header_id|>system<|end_header_id|>");
            println!("   {{system_message}}<|eot_id|>");
            println!("   <|start_header_id|>user<|end_header_id|>");
            println!("   {{user_message}}<|eot_id|>");
            println!("   <|start_header_id|>assistant<|end_header_id|>");
            println!("   {{assistant_message}}<|eot_id|>");
            println!();
            println!("3. llama2:");
            println!("   [INST] <<SYS>>{{system_message}}<</SYS>>");
            println!("   {{user_message}} [/INST] {{assistant_message}} </s>");
            println!();
            println!("4. mistral:");
            println!("   <s>[INST] {{user_message}} [/INST] {{assistant_message}}</s>");
            println!();
            println!("5. phi:");
            println!("   <|system|>{{system_message}}<|end|>");
            println!("   <|user|>{{user_message}}<|end|>");
            println!("   <|assistant|>{{assistant_message}}<|end|>");
            println!();
            println!("6. gemma:");
            println!("   <start_of_turn>user");
            println!("   {{user_message}}<end_of_turn>");
            println!("   <start_of_turn>model");
            println!("   {{assistant_message}}<end_of_turn>");
            println!();
            println!("7. qwen:");
            println!("   Same as ChatML");
            println!();
            println!("8. raw:");
            println!("   No template, concatenates messages with newlines");
            println!();
            println!("9. auto:");
            println!("   Uses the tokenizer's built-in chat_template");
            println!();

            println!("EXAMPLE WORKFLOW:");
            println!("-----------------");
            println!("# Download and preview");
            println!("pmetal dataset preview tatsu-lab/alpaca --num 3");
            println!();
            println!("# Full preparation pipeline");
            println!("pmetal dataset prepare tatsu-lab/alpaca \\");
            println!("  --output-dir ./alpaca_prepared \\");
            println!("  --model unsloth/Qwen3-0.6B \\");
            println!("  --template chatml \\");
            println!("  --max-seq-len 2048 \\");
            println!("  --val-ratio 0.05");
            println!();
            println!("# Or step by step:");
            println!("pmetal dataset download tatsu-lab/alpaca -o raw.jsonl");
            println!(
                "pmetal dataset filter -i raw.jsonl -o filtered.jsonl --model ... --max-tokens 2048 --dedup"
            );
            println!(
                "pmetal dataset template -i filtered.jsonl -o templated.jsonl --template chatml"
            );
            println!("pmetal dataset split -i templated.jsonl -o ./splits --val-ratio 0.1");
            println!();
            println!("# Analyze your data");
            println!("pmetal dataset analyze -p train.jsonl --model unsloth/Qwen3-0.6B");
        }
    }

    Ok(())
}

/// Extract text content from a sample for filtering/analysis.
fn extract_text_from_sample(json: &serde_json::Value) -> String {
    if let Some(t) = json.get("text").and_then(|v| v.as_str()) {
        t.to_string()
    } else if let Some(convs) = json.get("conversations").and_then(|v| v.as_array()) {
        convs
            .iter()
            .filter_map(|c| c.get("value").and_then(|v| v.as_str()))
            .collect::<Vec<_>>()
            .join(" ")
    } else if let Some(inst) = json.get("instruction").and_then(|v| v.as_str()) {
        let input = json.get("input").and_then(|v| v.as_str()).unwrap_or("");
        let output = json.get("output").and_then(|v| v.as_str()).unwrap_or("");
        format!("{} {} {}", inst, input, output)
    } else if let Some(msgs) = json.get("messages").and_then(|v| v.as_array()) {
        msgs.iter()
            .filter_map(|m| m.get("content").and_then(|v| v.as_str()))
            .collect::<Vec<_>>()
            .join(" ")
    } else {
        String::new()
    }
}

/// Format conversations with a chat template.
fn format_conversations(
    template: &ChatTemplate,
    conversations: &[(String, String)],
    system_msg: Option<&str>,
    add_generation_prompt: bool,
) -> String {
    let mut output = String::new();

    match template {
        ChatTemplate::Chatml | ChatTemplate::Qwen => {
            if let Some(sys) = system_msg {
                output.push_str(&format!("<|im_start|>system\n{}<|im_end|>\n", sys));
            }
            for (role, content) in conversations {
                let role_name = match role.as_str() {
                    "human" | "user" => "user",
                    "gpt" | "assistant" => "assistant",
                    "system" => "system",
                    _ => role.as_str(),
                };
                output.push_str(&format!(
                    "<|im_start|>{}\n{}<|im_end|>\n",
                    role_name, content
                ));
            }
            if add_generation_prompt {
                output.push_str("<|im_start|>assistant\n");
            }
        }

        ChatTemplate::Llama3 => {
            output.push_str("<|begin_of_text|>");
            if let Some(sys) = system_msg {
                output.push_str(&format!(
                    "<|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>",
                    sys
                ));
            }
            for (role, content) in conversations {
                let role_name = match role.as_str() {
                    "human" | "user" => "user",
                    "gpt" | "assistant" => "assistant",
                    "system" => "system",
                    _ => role.as_str(),
                };
                output.push_str(&format!(
                    "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
                    role_name, content
                ));
            }
            if add_generation_prompt {
                output.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
            }
        }

        ChatTemplate::Llama2 => {
            output.push_str("<s>");
            let mut first_user = true;
            for (role, content) in conversations {
                match role.as_str() {
                    "human" | "user" => {
                        if first_user {
                            if let Some(sys) = system_msg {
                                output.push_str(&format!(
                                    "[INST] <<SYS>>\n{}\n<</SYS>>\n\n{} [/INST] ",
                                    sys, content
                                ));
                            } else {
                                output.push_str(&format!("[INST] {} [/INST] ", content));
                            }
                            first_user = false;
                        } else {
                            output.push_str(&format!("<s>[INST] {} [/INST] ", content));
                        }
                    }
                    "gpt" | "assistant" => {
                        output.push_str(&format!("{} </s>", content));
                    }
                    _ => {}
                }
            }
            if add_generation_prompt {
                output.push_str("[INST] ");
            }
        }

        ChatTemplate::Mistral => {
            output.push_str("<s>");
            for (role, content) in conversations {
                match role.as_str() {
                    "human" | "user" => {
                        output.push_str(&format!("[INST] {} [/INST]", content));
                    }
                    "gpt" | "assistant" => {
                        output.push_str(&format!("{}</s>", content));
                    }
                    _ => {}
                }
            }
            if add_generation_prompt {
                output.push_str("[INST] ");
            }
        }

        ChatTemplate::Phi => {
            if let Some(sys) = system_msg {
                output.push_str(&format!("<|system|>\n{}<|end|>\n", sys));
            }
            for (role, content) in conversations {
                let role_name = match role.as_str() {
                    "human" | "user" => "user",
                    "gpt" | "assistant" => "assistant",
                    "system" => "system",
                    _ => role.as_str(),
                };
                output.push_str(&format!("<|{}|>\n{}<|end|>\n", role_name, content));
            }
            if add_generation_prompt {
                output.push_str("<|assistant|>\n");
            }
        }

        ChatTemplate::Gemma => {
            for (role, content) in conversations {
                let role_name = match role.as_str() {
                    "human" | "user" => "user",
                    "gpt" | "assistant" => "model",
                    _ => role.as_str(),
                };
                output.push_str(&format!(
                    "<start_of_turn>{}\n{}<end_of_turn>\n",
                    role_name, content
                ));
            }
            if add_generation_prompt {
                output.push_str("<start_of_turn>model\n");
            }
        }

        ChatTemplate::Raw => {
            for (_, content) in conversations {
                output.push_str(content);
                output.push('\n');
            }
        }

        ChatTemplate::Auto => {
            // For auto, we'd need to load the tokenizer's chat_template
            // For now, fall back to ChatML
            if let Some(sys) = system_msg {
                output.push_str(&format!("<|im_start|>system\n{}<|im_end|>\n", sys));
            }
            for (role, content) in conversations {
                let role_name = match role.as_str() {
                    "human" | "user" => "user",
                    "gpt" | "assistant" => "assistant",
                    _ => role.as_str(),
                };
                output.push_str(&format!(
                    "<|im_start|>{}\n{}<|im_end|>\n",
                    role_name, content
                ));
            }
            if add_generation_prompt {
                output.push_str("<|im_start|>assistant\n");
            }
        }
    }

    output
}
