//! PMetal CLI - LLM fine-tuning for Apple Silicon.

#![allow(clippy::too_many_arguments)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::needless_borrows_for_generic_args)]
#![allow(clippy::unnecessary_unwrap)]
#![allow(dead_code)] // Chat template formatters — pending migration to pmetal-data

pub mod cli;
mod commands;
mod dashboard;
#[cfg(any(feature = "models", feature = "native-only"))]
pub mod native_inference;
mod pack_experts;
#[cfg(feature = "dashboard")]
mod tui;

use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand, ValueEnum};
#[cfg(feature = "trainer")]
use pmetal_core::{LoraConfig, TrainingConfig};

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

/// GGUF quantization method for the `quantize` subcommand.
///
/// Supports all K-quant types available in the pmetal-gguf crate, plus
/// `dynamic` for importance-matrix-guided mixed-precision quantization.
/// K-quant size suffixes follow the GGUF K-quant naming convention:
///   - `s` = small (lower quality, smaller size)
///   - `m` = medium (balanced, recommended)
///   - `l` = large (higher quality, larger size)
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
pub enum QuantizeMethod {
    /// Importance-matrix-guided mixed precision (recommended with --imatrix)
    #[default]
    Dynamic,
    /// 8-bit integer (near-lossless, ~1.06 bpw)
    Q8_0,
    /// 6-bit K-quant (high quality, ~0.80 bpw)
    Q6K,
    /// 5-bit K-quant medium (good quality/size balance)
    Q5KM,
    /// 5-bit K-quant small (slightly smaller than q5-k-m)
    Q5KS,
    /// 4-bit K-quant medium (recommended 4-bit, ~0.58 bpw)
    Q4KM,
    /// 4-bit K-quant small (smallest 4-bit K-quant)
    Q4KS,
    /// 3-bit K-quant medium
    Q3KM,
    /// 3-bit K-quant small
    Q3KS,
    /// 3-bit K-quant large
    Q3KL,
    /// 2-bit K-quant (lowest quality, ~0.37 bpw)
    Q2K,
    /// 16-bit float (lossless, 2.0 bpw)
    F16,
    /// 32-bit float (lossless reference precision, 4.0 bpw)
    F32,
}

impl QuantizeMethod {
    /// Canonical display name matching the GGUF K-quant naming convention.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Dynamic => "dynamic",
            Self::Q8_0 => "q8_0",
            Self::Q6K => "q6_k",
            Self::Q5KM => "q5_k_m",
            Self::Q5KS => "q5_k_s",
            Self::Q4KM => "q4_k_m",
            Self::Q4KS => "q4_k_s",
            Self::Q3KM => "q3_k_m",
            Self::Q3KS => "q3_k_s",
            Self::Q3KL => "q3_k_l",
            Self::Q2K => "q2_k",
            Self::F16 => "f16",
            Self::F32 => "f32",
        }
    }

    /// Map to the underlying `GgmlType`. Returns `None` for `Dynamic` (uses
    /// `DynamicQuantizer` instead of a fixed type).
    pub fn to_ggml_type(self) -> Option<pmetal_gguf::GgmlType> {
        use pmetal_gguf::GgmlType;
        match self {
            Self::Dynamic => None,
            Self::Q8_0 => Some(GgmlType::Q8_0),
            Self::Q6K => Some(GgmlType::Q6K),
            // All Q5K size variants share the same K-quant block structure
            Self::Q5KM | Self::Q5KS => Some(GgmlType::Q5K),
            // All Q4K size variants share the same K-quant block structure
            Self::Q4KM | Self::Q4KS => Some(GgmlType::Q4K),
            // All Q3K size variants share the same K-quant block structure
            Self::Q3KM | Self::Q3KS | Self::Q3KL => Some(GgmlType::Q3K),
            Self::Q2K => Some(GgmlType::Q2K),
            Self::F16 => Some(GgmlType::F16),
            Self::F32 => Some(GgmlType::F32),
        }
    }
}

/// Inference-context selection for `bench-workload`.
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
pub enum WorkloadInferenceContext {
    /// Automatically choose between prompt-only and longer text-prefix continuation context
    #[default]
    Auto,
    /// Use the prompt portion only
    Prompt,
    /// Use a prefix of the full formatted sample text
    TextPrefix,
}

/// GDN projection stage for `bench-gdn`.
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
pub enum GdnBenchmarkStage {
    /// Benchmark the four GDN input projections (`qkv`, `z`, `b`, `a`)
    #[default]
    #[value(name = "input-proj")]
    InputProj,
    /// Benchmark the GDN output projection after recurrent update + norm
    #[value(name = "out-proj")]
    OutProj,
    /// Benchmark the recurrent GDN prefill update across fixed chunk sizes
    #[value(name = "prefill")]
    Prefill,
}

/// Named workload presets for `bench-workload`.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum WorkloadBenchmarkPreset {
    /// Dense Qwen3 regression workload on the cached reasoning dataset
    #[value(name = "dense-qwen3")]
    DenseQwen3,
    /// Hybrid Qwen3Next regression workload with a conservative training cap
    #[value(name = "hybrid-qwen3next")]
    HybridQwen3Next,
    /// Longer-running steady-state hybrid Qwen3.5 inference regression workload
    #[value(name = "hybrid-qwen35-steady")]
    HybridQwen35Steady,
    /// Nemotron-H sparse/hybrid inference regression workload
    #[value(name = "moe-nemotronh")]
    MoeNemotronH,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum TurboQuantPresetArg {
    #[value(name = "q2_5")]
    Q2_5,
    #[value(name = "q3_5")]
    Q3_5,
}

impl From<TurboQuantPresetArg> for pmetal::inference_runner::TurboQuantPreset {
    fn from(value: TurboQuantPresetArg) -> Self {
        match value {
            TurboQuantPresetArg::Q2_5 => Self::Q2_5,
            TurboQuantPresetArg::Q3_5 => Self::Q3_5,
        }
    }
}

#[derive(Parser)]
#[command(name = "pmetal")]
#[command(author, version, about = "LLM fine-tuning optimized for Apple Silicon", long_about = None)]
struct Cli {
    /// Write every JobEvent as a JSONL line to this file.
    ///
    /// When set, the CLI wraps each long-running job's callback chain with a
    /// `pmetal_core::JsonlSink<File>` so the event stream is machine-readable.
    /// This is what TUI subprocess-fallback and MCP subprocess consumers read.
    ///
    /// Currently wired for `pmetal train`. Other subcommands will follow the
    /// same pattern (see `crates/pmetal/src/cli/README.md`).
    #[cfg(feature = "trainer")]
    #[arg(long, global = true, value_name = "PATH")]
    log_events: Option<PathBuf>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
#[allow(clippy::large_enum_variant)]
enum Commands {
    /// Fine-tune a model using LoRA/QLoRA
    #[cfg(feature = "trainer")]
    Train(crate::cli::train::TrainArgs),

    /// Pretrain a model from scratch (full-parameter, no LoRA)
    #[cfg(feature = "trainer")]
    Pretrain(crate::cli::pretrain::PretrainArgs),

    /// Run inference with a model
    Infer(crate::cli::infer::InferArgs),


    /// Download a model from HuggingFace
    Download {
        /// Model ID
        model: String,

        /// Specific revision
        #[arg(long)]
        revision: Option<String>,
    },

    /// Search HuggingFace Hub for models and show device fit
    Search {
        /// Search query (e.g. "qwen3 0.6B", "llama 8b")
        query: String,

        /// Maximum number of results
        #[arg(short, long, default_value = "15")]
        limit: usize,

        /// Download the first result that fits on your device
        #[arg(long)]
        download: bool,

        /// Show detailed fit analysis (memory breakdown, training estimates)
        #[arg(long)]
        detailed: bool,

        /// Output results as JSON
        #[arg(long)]
        json: bool,
    },

    /// Block-diffusion speculative decoding (DFlash).
    ///
    /// Runs a Qwen3 target alongside a DFlash draft to emit multiple tokens
    /// per forward pass with bit-identical output to greedy baseline.
    Dflash(crate::cli::dflash::DflashArgs),

    /// Show memory usage and available capacity
    Memory {
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Benchmark FFI overhead (for performance analysis)
    #[cfg(feature = "trainer")]
    BenchFfi,

    /// Benchmark generation loop timing (detailed profiling)
    #[cfg(feature = "trainer")]
    BenchGen {
        /// Model to benchmark
        #[arg(short, long, default_value = "Qwen/Qwen3-0.6B")]
        model: String,
    },

    /// Benchmark training performance
    #[cfg(feature = "trainer")]
    Bench(crate::cli::bench::BenchArgs),

    /// Run a structured kernel benchmark corpus for this device tier
    #[cfg(feature = "trainer")]
    BenchCorpus {
        /// Use a shorter run with fewer iterations and smaller tier-scaled shapes
        #[arg(long)]
        quick: bool,

        /// Output results as JSON
        #[arg(long)]
        json: bool,

        /// Optional output path for the JSON report
        #[arg(short, long)]
        output: Option<String>,
    },

    /// Benchmark a real cached workload for inference and short LoRA training
    #[cfg(feature = "trainer")]
    BenchWorkload {
        /// Named preset that overrides the model/dataset/shape knobs below
        #[arg(long, value_enum)]
        preset: Option<WorkloadBenchmarkPreset>,

        /// Model ID or local path
        #[arg(long, default_value = "Qwen/Qwen3-0.6B")]
        model: String,

        /// Dataset ID or local path
        #[arg(
            long,
            default_value = "TeichAI/gemini-3-pro-preview-high-reasoning-250x"
        )]
        dataset: String,

        /// Packed expert directory for sparse/offloaded inference workloads
        #[arg(long)]
        experts_dir: Option<String>,

        /// Number of prompt samples to benchmark for inference
        #[arg(long, default_value = "8")]
        prompt_samples: usize,

        /// Maximum prompt tokens per inference sample (0 = auto-select from sampled prompts, capped for a quick run)
        #[arg(long, default_value = "0")]
        max_prompt_tokens: usize,

        /// Inference context source for workload benchmarking
        #[arg(long, value_enum, default_value = "auto")]
        inference_context: WorkloadInferenceContext,

        /// Number of decode steps per prompt sample
        #[arg(long, default_value = "32")]
        decode_steps: usize,

        /// Number of untimed warmup passes to run per sampled inference prompt before measurement
        #[arg(long, default_value = "2")]
        inference_warmup_passes: usize,

        /// Number of independent warmed inference sessions to run and aggregate
        #[arg(long, default_value = "1")]
        inference_session_repeats: usize,

        /// Number of timed inference passes per prompt sample
        #[arg(long, default_value = "1")]
        inference_repeats: usize,

        /// Number of raw samples to include in the sampled training subset
        #[arg(long, default_value = "8")]
        train_samples: usize,

        /// Number of LoRA training steps to run
        #[arg(long, default_value = "4")]
        train_steps: usize,

        /// Training batch size
        #[arg(long, default_value = "1")]
        batch_size: usize,

        /// Maximum sequence length for the short training run (0 = auto-select from sampled data, capped for a quick run)
        #[arg(long, default_value = "0")]
        max_seq_len: usize,

        /// Output results as JSON
        #[arg(long)]
        json: bool,

        /// Optional output path for the JSON report
        #[arg(short, long)]
        output: Option<String>,
    },

    /// Benchmark Qwen3.5 GDN backends on the actual model layer shapes
    #[cfg(feature = "trainer")]
    BenchGdn {
        /// Qwen3.5/Qwen3Next model ID or local path
        #[arg(long, default_value = "unsloth/Qwen3.5-0.8B")]
        model: String,

        /// Projection stage to benchmark
        #[arg(long, value_enum, default_value = "input-proj")]
        stage: GdnBenchmarkStage,

        /// Optional transformer layer index to benchmark (defaults to the first GDN layer)
        #[arg(long)]
        layer: Option<usize>,

        /// Batch size for the synthetic benchmark input
        #[arg(long, default_value = "1")]
        batch_size: usize,

        /// Sequence length for the synthetic input (use a longer value for prefill benchmarking)
        #[arg(long, default_value = "1")]
        seq_len: usize,

        /// Warmup iterations per backend
        #[arg(long, default_value = "10")]
        warmup_iterations: usize,

        /// Timed iterations per backend
        #[arg(long, default_value = "50")]
        benchmark_iterations: usize,

        /// Output results as JSON
        #[arg(long)]
        json: bool,

        /// Optional output path for the JSON report
        #[arg(short, long)]
        output: Option<String>,
    },

    /// Generate a sample configuration file
    #[cfg(feature = "trainer")]
    Init {
        /// Output path for the config file
        #[arg(short, long, default_value = "config.yaml")]
        output: String,
    },

    /// Pack expert weights for SSD-offloaded MoE inference
    PackExperts(crate::cli::pack_experts::PackExpertsArgs),

    /// Export trained model for Ollama
    Ollama {
        #[command(subcommand)]
        action: OllamaAction,
    },

    /// Fuse LoRA adapter weights into a base model and save as a complete model
    #[cfg(feature = "lora")]
    Fuse(crate::cli::fuse::FuseArgs),

    /// Quantize a model to GGUF format (supports Dynamic 2.0 and KL-calibrated)
    Quantize(crate::cli::quantize::QuantizeArgs),

    /// Knowledge Distillation from teacher to student
    #[cfg(feature = "trainer")]
    Distill(crate::cli::distill::DistillArgs),

    /// Group Relative Policy Optimization (GRPO) for reasoning models
    #[cfg(feature = "trainer")]
    Grpo(crate::cli::grpo::GrpoArgs),

    /// RLKD: Reinforcement Learning with Knowledge Distillation.
    ///
    /// Combines GRPO policy gradient optimization with knowledge distillation
    /// from a teacher model in a single training loop.
    ///
    /// Loss formula: L = (1 - alpha) * L_grpo + alpha * L_distill
    #[cfg(feature = "trainer")]
    Rlkd(crate::cli::rlkd::RlkdArgs),

    /// Start MCP server for Claude Desktop integration
    #[cfg(feature = "mcp")]
    Mcp,

    /// Start an OpenAI-compatible inference server
    #[cfg(feature = "serve")]
    Serve(crate::cli::serve::ServeArgs),

    /// Dataset utilities for preparing and analyzing training data
    Dataset {
        #[command(subcommand)]
        action: DatasetAction,
    },

    /// Tokenize a text corpus into binary shards for pretraining
    Tokenize(crate::cli::tokenize::TokenizeArgs),

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

    /// Show device information (GPU architecture, ANE cores, bandwidth, NAX support)
    Info {
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Merge two or more models using various merge methods (SLERP, TIES, DARE, linear, etc.)
    #[cfg(feature = "merge")]
    Merge(crate::cli::merge::MergeArgs),

    /// Evaluate a model's perplexity on a dataset
    #[cfg(feature = "lora")]
    Eval(crate::cli::eval::EvalArgs),

    /// Train a sentence embedding model (BERT / encoder-only) with contrastive losses.
    ///
    /// Supports InfoNCE (default), triplet, CoSENT, and cosine-similarity losses.
    /// Input data must be JSONL with pair or triplet format.
    ///
    /// Pair JSONL:
    ///   {"text_a": "...", "text_b": "...", "label": 0.9}
    ///   {"sentence1": "...", "sentence2": "..."}
    ///   {"query": "...", "positive": "..."}
    ///
    /// Triplet JSONL:
    ///   {"anchor": "...", "positive": "...", "negative": "..."}
    #[command(name = "embed-train")]
    #[cfg(feature = "trainer")]
    EmbedTrain(crate::cli::embed_train::EmbedTrainArgs),
}

/// Dataset subcommands for data preparation.
#[derive(Subcommand, Debug)]
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

        /// Column mapping: remap source columns to standard names.
        /// Format: "target=source,target=source" (e.g., "instruction=problem,output=solution")
        #[arg(long)]
        columns: Option<String>,
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
#[derive(Subcommand, Debug)]
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

/// Gzip-compressed `mlx.metallib` bytes (baked in at compile time by build.rs).
/// ~31MB compressed in the binary, ~102MB decompressed on disk.
#[cfg(target_os = "macos")]
static MLX_METALLIB_GZ: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mlx.metallib.gz"));

/// Extract the embedded `mlx.metallib` to `~/.cache/pmetal/lib/mlx.metallib`,
/// decompressing the gzip'd blob on first run. Skips if a file of non-zero
/// size already exists at the destination.
/// Returns the path if extraction succeeded.
#[cfg(target_os = "macos")]
fn extract_embedded_metallib() -> Option<PathBuf> {
    use flate2::read::GzDecoder;
    use std::io::Read;

    if MLX_METALLIB_GZ.is_empty() {
        return None;
    }
    let home = std::env::var("HOME").ok()?;
    let cache_dir = PathBuf::from(home).join(".cache/pmetal/lib");
    let dest = cache_dir.join("mlx.metallib");

    // Skip if already present with non-zero size
    if dest.is_file() {
        if let Ok(meta) = dest.metadata() {
            if meta.len() > 0 {
                return Some(dest);
            }
        }
    }

    std::fs::create_dir_all(&cache_dir).ok()?;
    let tmp = dest.with_extension("metallib.tmp");

    let mut decoder = GzDecoder::new(MLX_METALLIB_GZ);
    let mut decompressed = Vec::new();
    if decoder.read_to_end(&mut decompressed).is_err() {
        return None;
    }

    std::fs::write(&tmp, &decompressed).ok()?;
    std::fs::rename(&tmp, &dest).ok()?;
    tracing::info!("Extracted embedded mlx.metallib to {}", dest.display());
    Some(dest)
}

/// Discover `mlx.metallib` and set `PMETAL_METALLIB_PATH` so the patched MLX
/// C++ backend can find it regardless of where the binary is installed.
///
/// Search order: colocated → build dir → cache → Homebrew → embedded → download → error.
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

    // Try extracting the embedded metallib (baked in at compile time)
    if let Some(path) = extract_embedded_metallib() {
        unsafe {
            std::env::set_var("PMETAL_METALLIB_PATH", &path);
            std::env::set_var("MLX_METAL_JIT", "1");
        };
        tracing::debug!(path = %path.display(), "Using embedded mlx.metallib");
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
         \x1b[1m  1. Rebuild from source:\x1b[0m  cargo install pmetal  (auto-caches metallib)\n\
         \x1b[1m  2. Download manually:\x1b[0m\n\
             curl -fSL -o ~/.cache/pmetal/lib/mlx.metallib \\\n\
               https://github.com/epistates/pmetal/releases/download/v{version}/mlx.metallib\n\
         \x1b[1m  3. Homebrew:\x1b[0m             brew install epistates/tap/pmetal\n\n\
         The metallib ships with every pmetal release and is built during compilation.\n\
         See: https://github.com/epistates/pmetal#installation\n",
        version = env!("CARGO_PKG_VERSION"),
    );
}

/// SHA-256 of the released `mlx.metallib` artifact.
///
/// Empty string in dev builds.  Set to the hex digest of the artifact for
/// official releases so every download is cryptographically verified before
/// being committed to the cache.
const METALLIB_SHA256: &str = match option_env!("PMETAL_METALLIB_SHA256") {
    Some(s) => s,
    None => "",
};

/// Download `mlx.metallib` from the matching GitHub release.
///
/// Uses `reqwest` (no subprocess) and verifies the SHA-256 of the downloaded
/// bytes when `METALLIB_SHA256` is non-empty.  Writes to a temp file first,
/// then atomically renames into place so a failed download never leaves a
/// corrupted cache entry.
///
/// Returns `true` if the download succeeded and the file is verified.
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

    // Use a temp file so a partial download never leaves a broken metallib.
    let tmp = dest.with_extension("metallib.tmp");

    // Build a single-threaded tokio runtime for the blocking download
    // (this function is called before the main async runtime is started).
    let rt = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(r) => r,
        Err(e) => {
            eprintln!("\x1b[1;33mwarning:\x1b[0m Failed to create download runtime: {e}");
            return false;
        }
    };

    let bytes: Vec<u8> = match rt.block_on(async {
        let client = reqwest::Client::builder()
            .user_agent(concat!("pmetal/", env!("CARGO_PKG_VERSION")))
            .build()?;
        let response = client
            .get(&url)
            .send()
            .await?
            .error_for_status()?
            .bytes()
            .await?;
        Ok::<_, reqwest::Error>(response.to_vec())
    }) {
        Ok(b) => b,
        Err(e) => {
            eprintln!(
                "\x1b[1;33mwarning:\x1b[0m Download failed (offline or release v{version} \
                 not published yet): {e}\n"
            );
            return false;
        }
    };

    // Verify SHA-256 when a reference digest is compiled in.
    if !METALLIB_SHA256.is_empty() {
        use sha2::Digest as _;
        let digest = sha2::Sha256::digest(&bytes);
        let hex = format!("{digest:x}");
        if hex != METALLIB_SHA256 {
            eprintln!(
                "\x1b[1;31merror:\x1b[0m SHA-256 mismatch for mlx.metallib\n\
                 expected: {METALLIB_SHA256}\n\
                 got:      {hex}\n\
                 The downloaded file has been discarded."
            );
            return false;
        }
    }

    // Write to temp file, then atomically rename into place.
    if let Err(e) = std::fs::write(&tmp, &bytes) {
        eprintln!("\x1b[1;33mwarning:\x1b[0m Failed to write temp metallib: {e}");
        return false;
    }

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

/// Synchronous entry point.
///
/// `ensure_metallib` calls `std::env::set_var` which is unsound when other
/// threads are running.  By invoking it here — before the `#[tokio::main]`
/// macro starts the async runtime and its thread pool — we guarantee that
/// the environment mutation happens in a single-threaded context.
fn main() -> anyhow::Result<()> {
    ensure_metallib();
    #[cfg(feature = "metal")]
    let _ = pmetal_metal::context::MetalContext::device_available();
    tokio_main()
}

// ---------------------------------------------------------------------------
// Persistent logging
// ---------------------------------------------------------------------------

/// Initialize logging with file output to `~/.cache/pmetal/logs/{component}.log`.
///
/// - **CLI mode**: logs to both stderr and file
/// - **TUI mode**: logs to file only (stderr would corrupt the terminal), or
///   to `PMETAL_LOG_FILE` if set for backwards compatibility
///
/// The previous log is rotated to `{component}.log.1`.
fn init_logging(component: &str, suppress_stderr: bool) {
    // Determine log file path
    let log_path = if let Ok(custom) = std::env::var("PMETAL_LOG_FILE") {
        std::path::PathBuf::from(custom)
    } else {
        let dir = pmetal_log_dir();
        let path = dir.join(format!("{component}.log"));

        // Rotate: keep one previous log
        let prev = dir.join(format!("{component}.log.1"));
        if path.exists() {
            let _ = std::fs::rename(&path, &prev);
        }
        path
    };

    let filter = tracing_subscriber::EnvFilter::from_default_env()
        .add_directive(tracing::Level::INFO.into());

    if suppress_stderr {
        // TUI: file only (no stderr to avoid corrupting terminal)
        if let Ok(file) = std::fs::File::create(&log_path) {
            tracing_subscriber::fmt()
                .with_writer(std::sync::Mutex::new(file))
                .with_env_filter(filter)
                .with_ansi(false)
                .init();
        }
    } else {
        // CLI: stderr + tee to file
        let file = std::fs::File::create(&log_path).ok();
        let file = file.map(std::sync::Mutex::new).map(std::sync::Arc::new);
        let file_clone = file.clone();

        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_writer(move || TeeWriter {
                stderr: std::io::stderr(),
                file: file_clone.clone(),
            })
            .init();
    }
}

fn pmetal_log_dir() -> std::path::PathBuf {
    let dir = dirs::cache_dir()
        .unwrap_or_else(|| std::path::PathBuf::from(".cache"))
        .join("pmetal")
        .join("logs");
    let _ = std::fs::create_dir_all(&dir);
    dir
}

/// Writer that tees output to both stderr and a log file.
struct TeeWriter {
    stderr: std::io::Stderr,
    file: Option<std::sync::Arc<std::sync::Mutex<std::fs::File>>>,
}

impl std::io::Write for TeeWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.stderr.write(buf)?;
        if let Some(ref file) = self.file {
            if let Ok(mut f) = file.lock() {
                let _ = f.write_all(&buf[..n]);
            }
        }
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.stderr.flush()?;
        if let Some(ref file) = self.file {
            if let Ok(mut f) = file.lock() {
                let _ = f.flush();
            }
        }
        Ok(())
    }
}

#[tokio::main]
async fn tokio_main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Capture global flags before `cli` is moved into the `match` below.
    #[cfg(feature = "trainer")]
    let log_events_path = cli.log_events.clone();

    // Initialize logging — suppress stderr output when running the TUI
    // to avoid corrupting the raw terminal display.
    #[cfg(feature = "dashboard")]
    let is_tui = matches!(cli.command, Commands::Tui { .. });
    #[cfg(not(feature = "dashboard"))]
    let is_tui = false;
    init_logging(if is_tui { "tui" } else { "cli" }, is_tui);

    match cli.command {
        #[cfg(feature = "trainer")]
        Commands::Train(args) => {
            let crate::cli::train::TrainArgs {
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
                pack_max_seq_len,
                no_jit_compilation,
                no_gradient_checkpointing,
                gradient_checkpointing_layers,
                log_metrics,
                embedding_lr,
                loss_scale,
                warmup_steps,
                lr_schedule,
                weight_decay,
                seed,
                cut_cross_entropy,
                no_adaptive_lr,
                text_column,
                text_columns,
                column_separator,
                prompt_column,
                response_column,
                #[cfg(feature = "ane")]
                ane,
                #[cfg(feature = "distributed")]
                distributed_peers,
                #[cfg(feature = "distributed")]
                distributed_auto,
                #[cfg(feature = "distributed")]
                compression_strategy,
            } = args;
            use pmetal_trainer::orchestrator;

            // Parse LR schedule
            let lr_scheduler = match lr_schedule.to_lowercase().as_str() {
                "constant" => pmetal_core::LrSchedulerType::Constant,
                "linear" => pmetal_core::LrSchedulerType::Linear,
                "cosine" => pmetal_core::LrSchedulerType::Cosine,
                "cosine_with_restarts" => pmetal_core::LrSchedulerType::CosineWithRestarts,
                "polynomial" => pmetal_core::LrSchedulerType::Polynomial,
                "wsd" => pmetal_core::LrSchedulerType::Wsd,
                other => {
                    anyhow::bail!(
                        "Unknown lr-schedule '{}'. Valid: constant, linear, cosine, cosine_with_restarts, polynomial, wsd",
                        other
                    );
                }
            };

            // Build QLoRA config if quantization is requested
            let qlora = if !matches!(quantization, QuantizationMethod::None) {
                Some(orchestrator::QLoraOrchConfig {
                    scheme: match quantization {
                        QuantizationMethod::Nf4 => orchestrator::QuantizationScheme::Nf4,
                        QuantizationMethod::Fp4 => orchestrator::QuantizationScheme::Fp4,
                        QuantizationMethod::Int8 => orchestrator::QuantizationScheme::Int8,
                        QuantizationMethod::None => unreachable!(),
                    },
                    block_size: quant_block_size,
                    double_quant,
                })
            } else {
                None
            };

            // Build column config
            let columns = commands::build_column_config(
                text_column,
                text_columns,
                column_separator,
                prompt_column,
                response_column,
            );

            // Build distributed config
            #[cfg(feature = "distributed")]
            let distributed = {
                let has_peers = distributed_peers.as_ref().is_some_and(|p| !p.is_empty());
                if has_peers || distributed_auto {
                    let compression = match compression_strategy.as_deref() {
                        Some("topk") => pmetal_core::DistributedCompression::TopK,
                        Some("fp16") => pmetal_core::DistributedCompression::Fp16,
                        Some("random") => pmetal_core::DistributedCompression::Random,
                        _ => pmetal_core::DistributedCompression::None,
                    };
                    Some(pmetal_core::DistributedTrainingConfig {
                        peers: distributed_peers.unwrap_or_default(),
                        auto_discover: distributed_auto,
                        compression,
                        ..Default::default()
                    })
                } else {
                    None
                }
            };

            let job_config = orchestrator::TrainingJobConfig {
                model_id: model.unwrap_or_default(),
                dataset: dataset.unwrap_or_default(),
                eval_dataset,
                output_dir: output,
                lora: LoraConfig {
                    r: lora_r,
                    alpha: lora_alpha,
                    ..Default::default()
                },
                qlora,
                training: TrainingConfig {
                    learning_rate,
                    batch_size,
                    num_epochs: epochs,
                    max_seq_len,
                    gradient_accumulation_steps,
                    max_grad_norm,
                    warmup_steps,
                    weight_decay,
                    lr_scheduler,
                    embedding_learning_rate: embedding_lr.map(|v| v as f64),
                    ..Default::default()
                },
                columns,
                dispatch: orchestrator::DispatchConfig {
                    flash_attention: !no_flash_attention,
                    sequence_packing: !no_sequence_packing,
                    pack_max_seq_len,
                    jit_compilation: !no_jit_compilation,
                    fused: !no_fused,
                    metal_fused_optimizer: !no_metal_fused_optimizer,
                    gradient_checkpointing: !no_gradient_checkpointing,
                    gradient_checkpointing_layers,
                    cut_cross_entropy,
                    no_adaptive_lr,
                    #[cfg(feature = "ane")]
                    ane,
                    #[cfg(not(feature = "ane"))]
                    ane: false,
                    loss_scale,
                    #[cfg(feature = "distributed")]
                    distributed,
                },
                config_path: config,
                log_metrics,
                resume,
                seed,
                emit_console_output: true,
            };

            // If --log-events <path> was given, open the file and wrap the
            // trainer callback chain with a JsonlSink so every JobEvent is
            // emitted as a JSONL line.  The file is created (or truncated) at
            // parse time so callers can `tail -f` it immediately.
            //
            // Pattern for other entry points (distill, grpo, …): open the
            // sink the same way, box it as `Box<dyn pmetal_core::TrainingCallback>`,
            // and push it into the `extra_callbacks` vec that each run_* fn
            // already accepts.  See `src/cli/README.md` for the full pattern.
            // `mut` is needed post-substrate-merge when push() is called.
            #[allow(unused_mut)]
            let mut extra_callbacks: Vec<Box<dyn pmetal_core::TrainingCallback>> = Vec::new();
            if let Some(path) = log_events_path {
                // Validate the path is writable immediately so the user gets a
                // clear error before training starts rather than partway through.
                std::fs::File::create(&path).map_err(|e| {
                    anyhow::anyhow!("--log-events: could not create '{}': {e}", path.display())
                })?;
                // TODO(Phase 4 substrate adoption): once this worktree merges
                // `main` (commit ce2f770 carrying pmetal_core::events), replace
                // the file-create-and-drop above with the full sink wiring:
                //
                //   use pmetal_core::{JsonlSink, TrainingCallbackToSink};
                //   let file = std::fs::File::create(&path)?;
                //   let sink = JsonlSink::new(file);
                //   extra_callbacks.push(Box::new(TrainingCallbackToSink::new(
                //       job_config.model_id.clone(),
                //       sink,
                //   )));
                //
                // See `src/cli/README.md` §Wiring --log-events for the pattern.
                tracing::warn!(
                    "--log-events stub: file '{}' created but event streaming \
                     requires merging the substrate branch (ce2f770). \
                     Events will not be written until the merge is complete.",
                    path.display()
                );
            }

            orchestrator::run_training(job_config, None, extra_callbacks).await?;
        }

        #[cfg(feature = "mcp")]
        Commands::Mcp => {
            pmetal_mcp::run_stdio()
                .await
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        }

        #[cfg(feature = "serve")]
        Commands::Serve(args) => {
            let crate::cli::serve::ServeArgs {
                model,
                port,
                host,
                max_seq_len,
                experts_dir,
                fp8,
                kv_quant,
                no_kv_quant,
                kv_group_size,
                kv_turboquant,
                kv_turboquant_preset,
                #[cfg(feature = "ane")]
                ane,
                #[cfg(feature = "ane")]
                ane_max_seq_len,
                #[cfg(feature = "ane")]
                ane_real_time,
                continuous_batch,
                cb_max_slots,
                cb_max_queue_depth,
            } = args;
            #[cfg(feature = "ane")]
            let ane_enabled = ane;
            #[cfg(not(feature = "ane"))]
            let ane_enabled = false;

            #[cfg(feature = "ane")]
            let serve_ane_max_seq_len = ane_max_seq_len;
            #[cfg(not(feature = "ane"))]
            let serve_ane_max_seq_len = 1024;
            #[cfg(feature = "ane")]
            let serve_ane_real_time = ane_real_time;
            #[cfg(not(feature = "ane"))]
            let serve_ane_real_time = false;

            commands::serve::run_serve(
                model,
                port,
                host,
                max_seq_len,
                experts_dir,
                fp8,
                kv_quant,
                no_kv_quant,
                kv_group_size,
                kv_turboquant,
                kv_turboquant_preset,
                ane_enabled,
                serve_ane_max_seq_len,
                serve_ane_real_time,
                continuous_batch,
                cb_max_slots,
                cb_max_queue_depth,
            )
            .await?;
        }

        Commands::Infer(args) => {
            let crate::cli::infer::InferArgs {
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
                mode,
                backend,
                draft_model,
                metal_sampler,
                compiled,
                stream,
                minimal,
                hide_thinking,
                tools,
                fp8,
                experts_dir,
                #[cfg(feature = "ane")]
                ane,
                #[cfg(feature = "ane")]
                ane_max_seq_len,
                #[cfg(feature = "ane")]
                ane_real_time,
                benchmark,
                benchmark_iters,
                benchmark_prompt_tokens,
                profile_layers,
                profile_output,
                kv_quant,
                kv_k_bits,
                kv_v_bits,
                kv_group_size,
                kv_turboquant,
                kv_turboquant_preset,
                kv_quant_preset,
                no_kv_quant,
                kv_qjl,
                detect_repetition,
            } = args;
            // Load tool definitions if provided
            let tool_defs: Option<Vec<pmetal_data::chat_templates::ToolDefinition>> =
                if let Some(ref tools_path) = tools {
                    let tools_json = std::fs::read_to_string(tools_path)
                        .map_err(|e| anyhow::anyhow!("Failed to read tools file: {}", e))?;
                    let defs: Vec<pmetal_data::chat_templates::ToolDefinition> =
                        serde_json::from_str(&tools_json)
                            .map_err(|e| anyhow::anyhow!("Failed to parse tools JSON: {}", e))?;
                    tracing::info!("Loaded {} tool definitions from {}", defs.len(), tools_path);
                    Some(defs)
                } else {
                    None
                };

            let validated_profile_output = profile_output
                .as_deref()
                .map(|path| validate_output_path(path, "infer profile output"))
                .transpose()?;

            commands::infer::run_inference(
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
                mode,
                backend,
                draft_model.as_deref(),
                metal_sampler,
                compiled,
                stream,
                minimal,
                hide_thinking,
                fp8,
                tool_defs.as_deref(),
                #[cfg(feature = "ane")]
                ane,
                #[cfg(not(feature = "ane"))]
                false,
                #[cfg(feature = "ane")]
                ane_max_seq_len,
                #[cfg(not(feature = "ane"))]
                1024,
                #[cfg(feature = "ane")]
                ane_real_time,
                #[cfg(not(feature = "ane"))]
                false,
                benchmark,
                benchmark_iters,
                benchmark_prompt_tokens,
                profile_layers,
                validated_profile_output.as_deref(),
                kv_quant,
                kv_k_bits,
                kv_v_bits,
                kv_group_size,
                kv_turboquant,
                kv_turboquant_preset.map(Into::into),
                kv_quant_preset,
                no_kv_quant,
                kv_qjl,
                detect_repetition,
                experts_dir.as_deref(),
            )
            .await?;
        }

        Commands::Download { model, revision } => {
            tracing::info!(model = %model, "Downloading model");
            let path = pmetal_hub::download_model(&model, revision.as_deref(), None).await?;
            println!("Model downloaded to: {}", path.display());
        }

        Commands::PackExperts(args) => {
            let crate::cli::pack_experts::PackExpertsArgs {
                model,
                output,
                bits,
            } = args;
            pack_experts::pack_experts(Path::new(&model), Path::new(&output), bits)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        }

        Commands::Search {
            query,
            limit,
            download,
            detailed,
            json,
        } => {
            commands::search::run_search(&query, limit, download, detailed, json).await?;
        }

        Commands::Dflash(args) => {
            let crate::cli::dflash::DflashArgs {
                target,
                draft,
                prompt,
                max_new_tokens,
                temperature,
                speculative_tokens,
                draft_fp8,
                json,
                no_chat,
                tree_budget,
            } = args;
            commands::dflash::run_dflash(
                &target,
                &draft,
                &prompt,
                max_new_tokens,
                temperature,
                speculative_tokens,
                draft_fp8,
                json,
                no_chat,
                tree_budget,
            )
            .await?;
        }

        Commands::Memory { json } => {
            let stats = pmetal_mlx::memory::get_memory_stats();
            if json {
                let obj = serde_json::json!({
                    "total_gb": stats.total_gb(),
                    "used_gb": stats.used_gb(),
                    "available_gb": stats.available_gb(),
                    "peak_gb": stats.peak_gb(),
                });
                println!("{}", serde_json::to_string_pretty(&obj)?);
            } else {
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
        }

        #[cfg(feature = "trainer")]
        Commands::Bench(args) => {
            let crate::cli::bench::BenchArgs {
                model,
                batch_size,
                seq_len,
            } = args;
            commands::bench::run_benchmark(&model, batch_size, seq_len).await?;
        }

        #[cfg(feature = "trainer")]
        Commands::BenchCorpus {
            quick,
            json,
            output,
        } => {
            let validated_output = output
                .as_deref()
                .map(|path| validate_output_path(path, "benchmark corpus output"))
                .transpose()?;
            commands::bench::run_kernel_benchmark_corpus(quick, validated_output.as_deref(), json)?;
        }

        #[cfg(feature = "trainer")]
        Commands::BenchWorkload {
            preset,
            model,
            dataset,
            experts_dir,
            prompt_samples,
            max_prompt_tokens,
            inference_context,
            decode_steps,
            inference_warmup_passes,
            inference_session_repeats,
            inference_repeats,
            train_samples,
            train_steps,
            batch_size,
            max_seq_len,
            json,
            output,
        } => {
            let validated_output = output
                .as_deref()
                .map(|path| validate_output_path(path, "workload benchmark output"))
                .transpose()?;
            if let Some(preset) = preset {
                commands::bench::run_workload_benchmark_preset(
                    preset,
                    inference_warmup_passes,
                    inference_session_repeats,
                    experts_dir.as_deref(),
                    validated_output.as_deref(),
                    json,
                )
                .await?;
            } else {
                commands::bench::run_workload_benchmark(
                    &model,
                    &dataset,
                    experts_dir.as_deref(),
                    prompt_samples,
                    max_prompt_tokens,
                    inference_context,
                    decode_steps,
                    inference_warmup_passes,
                    inference_session_repeats,
                    inference_repeats,
                    train_samples,
                    train_steps,
                    batch_size,
                    max_seq_len,
                    validated_output.as_deref(),
                    json,
                )
                .await?;
            }
        }

        #[cfg(feature = "trainer")]
        Commands::BenchGdn {
            model,
            stage,
            layer,
            batch_size,
            seq_len,
            warmup_iterations,
            benchmark_iterations,
            json,
            output,
        } => {
            let validated_output = output
                .as_deref()
                .map(|path| validate_output_path(path, "GDN benchmark output"))
                .transpose()?;
            commands::bench::run_gdn_decode_benchmark(
                &model,
                stage,
                layer,
                batch_size,
                seq_len,
                warmup_iterations,
                benchmark_iterations,
                validated_output.as_deref(),
                json,
            )
            .await?;
        }

        #[cfg(feature = "trainer")]
        Commands::Init { output } => {
            // Validate output path to prevent path traversal attacks
            let validated_output = validate_output_path(&output, "config output")?;
            commands::bench::generate_sample_config(&validated_output.to_string_lossy())?;
        }

        #[cfg(feature = "trainer")]
        Commands::BenchFfi => {
            commands::bench::run_ffi_benchmark()?;
        }

        #[cfg(feature = "trainer")]
        Commands::BenchGen { model } => {
            commands::bench::run_gen_benchmark(&model).await?;
        }

        Commands::Ollama { action } => {
            commands::ollama::run_ollama_command(action).await?;
        }

        #[cfg(feature = "lora")]
        Commands::Fuse(args) => {
            let crate::cli::fuse::FuseArgs {
                model,
                lora,
                output,
                alpha,
                rank,
                accurate,
                low_memory,
            } = args;
            if accurate {
                commands::fuse::run_fuse_accurate(&model, &lora, &output, low_memory).await?;
            } else {
                commands::fuse::run_fuse(&model, &lora, &output, alpha, rank).await?;
            }
        }

        Commands::Quantize(args) => {
            let crate::cli::quantize::QuantizeArgs {
                model,
                output,
                imatrix,
                method,
                lora,
                kl_calibrate,
                target_bpw,
                kl_threshold,
                format,
                bits,
                group_size,
            } = args;
            if format == "mlx" {
                // MLX safetensors path — no LoRA fusion support in this path yet.
                if lora.is_some() {
                    anyhow::bail!(
                        "--lora is not yet supported with --format mlx; \
                         fuse the adapter first with `pmetal fuse` then quantize"
                    );
                }
                commands::quantize::run_quantization_mlx(
                    &model, &output, bits, group_size, target_bpw,
                )
                .await?;
            } else {
                #[cfg(feature = "lora")]
                if let Some(lora_path) = &lora {
                    // Fuse LoRA first, then quantize the fused model
                    let fused_dir = format!("{output}.fused_tmp");
                    commands::fuse::run_fuse(&model, lora_path, &fused_dir, None, None).await?;
                    commands::quantize::run_quantization(
                        &fused_dir,
                        &output,
                        imatrix.as_deref(),
                        method,
                        kl_calibrate,
                        target_bpw,
                        kl_threshold,
                    )
                    .await?;
                    // Clean up temp dir
                    let _ = std::fs::remove_dir_all(&fused_dir);
                } else {
                    commands::quantize::run_quantization(
                        &model,
                        &output,
                        imatrix.as_deref(),
                        method,
                        kl_calibrate,
                        target_bpw,
                        kl_threshold,
                    )
                    .await?;
                }

                #[cfg(not(feature = "lora"))]
                {
                    if lora.is_some() {
                        anyhow::bail!(
                            "LoRA fusion during quantization requires the `lora` feature"
                        );
                    }
                    commands::quantize::run_quantization(
                        &model,
                        &output,
                        imatrix.as_deref(),
                        method,
                        kl_calibrate,
                        target_bpw,
                        kl_threshold,
                    )
                    .await?;
                }
            }
        }

        #[cfg(feature = "trainer")]
        Commands::Distill(args) => {
            let crate::cli::distill::DistillArgs {
                teacher,
                student,
                dataset,
                output,
                method,
                offline,
                offline_cache,
                offline_generate,
                offline_compression,
                offline_top_k,
                loss_type,
                temperature,
                alpha,
                rationale,
                rationale_weight,
                lora_r,
                lora_alpha,
                learning_rate,
                batch_size,
                epochs,
                max_seq_len,
                seed,
                text_column,
                text_columns,
                column_separator,
                prompt_column,
                response_column,
                log_metrics,
            } = args;
            commands::distill::run_distillation_cli(
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
                lora_alpha,
                learning_rate,
                batch_size,
                epochs,
                max_seq_len,
                seed,
                log_metrics,
                true,
                Vec::new(),
                text_column,
                text_columns,
                column_separator,
                prompt_column,
                response_column,
                commands::distill::OfflineCliOptions {
                    enabled: offline,
                    cache_path: offline_cache,
                    generate: offline_generate,
                    compression: offline_compression,
                    top_k: offline_top_k,
                },
            )
            .await?;
        }

        #[cfg(feature = "trainer")]
        Commands::Grpo(args) => {
            let crate::cli::grpo::GrpoArgs {
                model,
                dataset,
                output,
                num_generations,
                beta,
                learning_rate,
                epochs,
                lora_r,
                lora_alpha,
                max_seq_len,
                max_completion_length,
                seed,
                dapo,
                reasoning_rewards,
                no_flash_attention,
                vlm,
                max_image_size,
                reward_model,
                reward_model_max_length,
                reward_model_weight,
                reward_model_template,
                async_rewards,
                speculative,
                speculative_draft_tokens,
                grpo_kv_bits,
                text_column,
                text_columns,
                column_separator,
                prompt_column,
                response_column,
                log_metrics,
            } = args;
            commands::grpo::run_grpo_cli(
                &model,
                &dataset,
                &output,
                num_generations,
                beta,
                learning_rate,
                epochs,
                lora_r,
                lora_alpha,
                max_seq_len,
                max_completion_length,
                seed,
                dapo,
                reasoning_rewards,
                !no_flash_attention,
                vlm,
                max_image_size,
                reward_model,
                reward_model_max_length,
                reward_model_weight,
                reward_model_template,
                async_rewards,
                speculative,
                speculative_draft_tokens,
                grpo_kv_bits,
                log_metrics,
                true,
                Vec::new(),
                text_column,
                text_columns,
                column_separator,
                prompt_column,
                response_column,
            )
            .await?;
        }

        #[cfg(feature = "trainer")]
        Commands::Rlkd(args) => {
            let crate::cli::rlkd::RlkdArgs {
                model,
                teacher_model,
                dataset,
                output,
                distill_alpha,
                final_alpha,
                anneal_alpha,
                distill_temperature,
                num_generations,
                beta,
                learning_rate,
                epochs,
                lora_r,
                lora_alpha,
                max_seq_len,
                max_completion_length,
                seed,
                reasoning_rewards,
                no_flash_attention,
                text_column,
                text_columns,
                column_separator,
                prompt_column,
                response_column,
                log_metrics,
            } = args;
            commands::rlkd::run_rlkd_cli(
                &model,
                &teacher_model,
                &dataset,
                &output,
                distill_alpha,
                final_alpha,
                anneal_alpha,
                distill_temperature,
                num_generations,
                beta,
                learning_rate,
                epochs,
                lora_r,
                lora_alpha,
                max_seq_len,
                max_completion_length,
                seed,
                reasoning_rewards,
                !no_flash_attention,
                log_metrics,
                true,
                Vec::new(),
                text_column,
                text_columns,
                column_separator,
                prompt_column,
                response_column,
            )
            .await?;
        }

        Commands::Dataset { action } => {
            commands::dataset::run_dataset_command(action).await?;
        }

        Commands::Tokenize(args) => {
            let crate::cli::tokenize::TokenizeArgs {
                input,
                output,
                tokenizer,
                text_column,
                docs_per_shard,
            } = args;
            commands::tokenize::run_tokenize(
                &input,
                &output,
                &tokenizer,
                &text_column,
                docs_per_shard,
            )
            .await?;
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

        Commands::Info { json } => {
            commands::info::run_info(json).await?;
        }

        #[cfg(feature = "merge")]
        Commands::Merge(args) => {
            let crate::cli::merge::MergeArgs {
                model_a,
                model_b,
                output,
                method,
                base,
                t,
                weight_a,
                weight_b,
                density,
                dtype,
            } = args;
            commands::merge::run_merge_command(
                &model_a,
                &model_b,
                &output,
                &method,
                base.as_deref(),
                t,
                weight_a,
                weight_b,
                density,
                &dtype,
            )
            .await?;
        }

        #[cfg(feature = "lora")]
        Commands::Eval(args) => {
            let crate::cli::eval::EvalArgs {
                model,
                dataset,
                lora,
                max_seq_len,
                num_samples,
                json,
            } = args;
            commands::eval::run_eval(
                &model,
                &dataset,
                lora.as_deref(),
                max_seq_len,
                num_samples,
                json,
            )
            .await?;
        }

        #[cfg(feature = "trainer")]
        Commands::EmbedTrain(args) => {
            let crate::cli::embed_train::EmbedTrainArgs {
                model,
                dataset,
                output,
                loss,
                pooling,
                temperature,
                margin,
                learning_rate,
                batch_size,
                epochs,
                max_seq_len,
                weight_decay,
                no_normalize,
                log_every,
                seed,
            } = args;
            commands::embed_train::run_embed_train(
                &model,
                &dataset,
                &output,
                &loss,
                &pooling,
                temperature,
                margin,
                learning_rate,
                batch_size,
                epochs,
                max_seq_len,
                weight_decay,
                !no_normalize,
                log_every,
                seed,
            )
            .await?;
        }

        #[cfg(feature = "trainer")]
        Commands::Pretrain(args) => {
            let crate::cli::pretrain::PretrainArgs {
                arch,
                shards,
                seq_len,
                batch_size,
                steps,
                learning_rate,
                min_lr,
                warmup_steps,
                lr_schedule,
                weight_decay,
                max_grad_norm,
                eos_token_id,
                output,
                checkpoint_every,
                resume,
                model_config: _model_config,
                z_loss,
                gradient_accumulation_steps,
                log_every,
                eval_every,
                eval_batches,
                seed,
            } = args;
            use pmetal_bridge::compat::random;
            use pmetal_data::streaming::{StreamConfig, StreamingShardReader};
            use pmetal_trainer::pretrain::{self, PretrainConfig};

            random::seed(seed);
            let _output_dir = validate_output_path(&output, "pretrain output")?;

            // Collect shard paths (each CLI value is a literal path)
            let shard_paths: Vec<std::path::PathBuf> =
                shards.iter().map(std::path::PathBuf::from).collect();
            anyhow::ensure!(!shard_paths.is_empty(), "no shard files specified");
            println!(
                "Pretraining on {} shards, {} steps",
                shard_paths.len(),
                steps
            );

            // Parse LR schedule
            let lr_sched = match lr_schedule.to_lowercase().as_str() {
                "constant" => pmetal_core::LrSchedulerType::Constant,
                "linear" => pmetal_core::LrSchedulerType::Linear,
                "cosine" => pmetal_core::LrSchedulerType::Cosine,
                other => anyhow::bail!("unknown lr-schedule: {other}"),
            };

            // Build model via factory (supports llama, qwen2, qwen3, gemma, gemma4, mistral, phi, gpt-oss)
            use pmetal_bridge::compat::module::ModuleParameters;
            let config_path = _model_config.as_deref().map(std::path::Path::new);
            let mut model = pretrain::create_model(&arch, config_path)?;
            let n_layers = pretrain::n_layers(&model);
            println!(
                "Architecture: {arch}, layers: {n_layers}, params: {}",
                model.num_parameters()
            );

            let config = PretrainConfig {
                num_steps: steps,
                learning_rate,
                min_lr,
                warmup_steps,
                lr_schedule: lr_sched,
                weight_decay,
                betas: (0.9, 0.95),
                eps: 1e-8,
                max_grad_norm: if max_grad_norm > 0.0 {
                    Some(max_grad_norm)
                } else {
                    None
                },
                ignore_index: None,
                z_loss_coef: if z_loss > 0.0 { Some(z_loss) } else { None },
                n_layers,
                apply_init: resume.is_none(),
                checkpoint_every: if checkpoint_every > 0 {
                    Some(checkpoint_every)
                } else {
                    None
                },
                checkpoint_dir: Some(std::path::PathBuf::from(&output)),
                gradient_accumulation_steps,
                log_every,
                eval_every,
                eval_batches,
            };

            // Resume from checkpoint if requested
            if let Some(ref ckpt_dir) = resume {
                let mut opt = pmetal_bridge::compat::optimizers::AdamWBuilder::new(learning_rate)
                    .weight_decay(weight_decay)
                    .build()?;
                let meta = pretrain::load_checkpoint(
                    std::path::Path::new(ckpt_dir),
                    &mut model,
                    &mut opt,
                )?;
                println!("Resumed from step {}, loss {:.4}", meta.step, meta.loss);
            }

            // Streaming data pipeline
            let stream_config = StreamConfig {
                shard_paths,
                seq_len,
                batch_size,
                eos_token_id,
                resume_from: None,
            };
            let reader = StreamingShardReader::new(stream_config)?;

            // Convert streaming batches to MLX arrays
            let batch_iter = reader.map(move |(batch, _pos)| {
                let flat: Vec<i32> = batch
                    .iter()
                    .flat_map(|seq| seq.iter().map(|&t| t as i32))
                    .collect();
                pmetal_bridge::InlineArray::from_i32_slice_shaped(
                    &flat,
                    &[batch.len() as i32, seq_len as i32],
                )
            });

            // Run
            let losses = pretrain::run_pretrain(&mut model, &config, batch_iter)?;

            // Print summary
            if !losses.is_empty() {
                let last = losses.last().copied().unwrap_or(0.0);
                println!(
                    "Pretraining complete: {} steps, final loss {:.4}",
                    losses.len(),
                    last,
                );
            }
        }
    }

    Ok(())
}

/// Save adapter config alongside LoRA weights.
fn save_adapter_config(
    lora_weights_path: &std::path::Path,
    r: usize,
    alpha: f32,
    target_modules: &[String],
    use_rslora: bool,
    base_model: Option<&str>,
) -> anyhow::Result<()> {
    let mut adapter_config = serde_json::json!({
        "r": r,
        "alpha": alpha,
        "target_modules": target_modules,
        "use_rslora": use_rslora,
    });
    if let Some(bm) = base_model {
        adapter_config["base_model"] = serde_json::Value::String(bm.to_string());
    }
    let config_path = lora_weights_path
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .join("adapter_config.json");
    std::fs::write(&config_path, serde_json::to_string_pretty(&adapter_config)?)?;
    tracing::info!("Saved adapter config to {:?}", config_path);
    Ok(())
}

fn validate_output_path(path: &str, context: &str) -> anyhow::Result<PathBuf> {
    use std::path::{Component, PathBuf};

    let path = PathBuf::from(path);

    for component in path.components() {
        if matches!(component, Component::ParentDir) {
            anyhow::bail!(
                "Path traversal detected in {}: '{}' contains '..' component. Please use a safe path.",
                context,
                path.display()
            );
        }
    }

    let cwd = std::env::current_dir()?;
    let resolved = if path.is_absolute() {
        path.clone()
    } else {
        cwd.join(&path)
    };

    if let Some(parent) = resolved.parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent)?;
        }
    }

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
        || canonical.starts_with("/tmp")
        || canonical.starts_with("/private/tmp");

    if !is_safe {
        anyhow::bail!(
            "Unsafe output path for {}: '{}' resolves to '{}'",
            context,
            path.display(),
            canonical.display()
        );
    }

    Ok(canonical)
}

// ---------------------------------------------------------------------------
// argv round-trip tests
//
// These verify that every JobSpec's `to_argv()` output is accepted by the
// actual clap `Commands` enum — i.e. the spec's `argv = "..."` attributes are
// byte-identical to the CLI flag names.
//
// The tests live here (not in `tests/`) because `Cli` is private to the binary.
// See `crates/pmetal/src/cli/README.md` for the future migration pattern that
// would move `Cli`/`Commands` to `lib.rs` and make integration tests possible.
//
// COMPILATION PREREQUISITE: These tests require `pmetal_core::jobs::*` which
// is added in commit ce2f770 (Phase 3 substrate).  They will not compile until
// this worktree is rebased/merged onto main.  The feature gate
// `feature = "core"` is necessary but not sufficient on its own — the crate
// must also have the jobs module.
//
// To run after the merge:
//   cargo test -p pmetal --features trainer -- argv_roundtrip
// ---------------------------------------------------------------------------
#[cfg(all(test, feature = "trainer", feature = "core"))]
mod argv_roundtrip {
    use super::{Cli, Commands};
    use clap::Parser;
    use pmetal_core::jobs::{
        BenchSpec, DflashSpec, DistillSpec, EmbedTrainSpec, EvalSpec, FuseSpec, GrpoSpec,
        InferSpec, MergeSpec, PackExpertsSpec, PretrainSpec, QuantizeSpec, RlkdSpec,
        TokenizeSpec, TrainSpec,
    };
    #[cfg(feature = "serve")]
    use pmetal_core::jobs::ServeSpec;
    use pmetal_core::JobFields;

    /// Parse an argv slice built from a spec's `to_argv()` output.  Returns
    /// `Err(String)` on failure so the test body can provide a descriptive message.
    fn try_parse(subcommand: &str, spec_argv: Vec<String>) -> Result<Commands, String> {
        let mut argv = vec!["pmetal".to_string(), subcommand.to_string()];
        argv.extend(spec_argv);
        Cli::try_parse_from(&argv)
            .map(|cli| cli.command)
            .map_err(|e| e.to_string())
    }

    #[test]
    fn train_spec_round_trip() {
        let mut spec = TrainSpec::default();
        spec.model = "Qwen/Qwen3-0.6B".into();
        spec.dataset = "data/train.jsonl".into();
        let result = try_parse("train", spec.to_argv());
        assert!(
            result.is_ok(),
            "TrainSpec argv failed to parse: {}",
            result.unwrap_err()
        );
    }

    #[test]
    fn distill_spec_round_trip() {
        let mut spec = DistillSpec::default();
        spec.teacher = "Qwen/Qwen3-7B".into();
        spec.student = "Qwen/Qwen3-0.6B".into();
        spec.dataset = "data.jsonl".into();
        let result = try_parse("distill", spec.to_argv());
        assert!(
            result.is_ok(),
            "DistillSpec argv failed to parse: {}",
            result.unwrap_err()
        );
    }

    #[test]
    fn grpo_spec_round_trip() {
        let mut spec = GrpoSpec::default();
        spec.model = "model".into();
        spec.dataset = "data.jsonl".into();
        let result = try_parse("grpo", spec.to_argv());
        assert!(
            result.is_ok(),
            "GrpoSpec argv failed to parse: {}",
            result.unwrap_err()
        );
    }

    #[test]
    fn rlkd_spec_round_trip() {
        let mut spec = RlkdSpec::default();
        spec.model = "model".into();
        spec.teacher_model = "teacher".into();
        spec.dataset = "data.jsonl".into();
        let result = try_parse("rlkd", spec.to_argv());
        assert!(
            result.is_ok(),
            "RlkdSpec argv failed to parse: {}",
            result.unwrap_err()
        );
    }

    #[test]
    fn embed_train_spec_round_trip() {
        let mut spec = EmbedTrainSpec::default();
        spec.model = "model".into();
        spec.dataset = "data.jsonl".into();
        let result = try_parse("embed-train", spec.to_argv());
        assert!(
            result.is_ok(),
            "EmbedTrainSpec argv failed to parse: {}",
            result.unwrap_err()
        );
    }

    #[test]
    fn pretrain_spec_round_trip() {
        let mut spec = PretrainSpec::default();
        spec.arch = "gpt-oss".into();
        let result = try_parse("pretrain", spec.to_argv());
        assert!(
            result.is_ok(),
            "PretrainSpec argv failed to parse: {}",
            result.unwrap_err()
        );
    }

    #[test]
    fn infer_spec_round_trip() {
        let mut spec = InferSpec::default();
        spec.model = "model".into();
        spec.prompt = "Hello".into();
        let result = try_parse("infer", spec.to_argv());
        assert!(
            result.is_ok(),
            "InferSpec argv failed to parse: {}",
            result.unwrap_err()
        );
    }

    // Commands::Serve is only compiled when `--features serve` is active.
    #[cfg(feature = "serve")]
    #[test]
    fn serve_spec_round_trip() {
        let mut spec = ServeSpec::default();
        spec.model = "model".into();
        let result = try_parse("serve", spec.to_argv());
        assert!(
            result.is_ok(),
            "ServeSpec argv failed to parse: {}",
            result.unwrap_err()
        );
    }

    #[test]
    fn bench_spec_round_trip() {
        let spec = BenchSpec::default();
        let result = try_parse("bench", spec.to_argv());
        assert!(
            result.is_ok(),
            "BenchSpec argv failed to parse: {}",
            result.unwrap_err()
        );
    }

    #[test]
    fn eval_spec_round_trip() {
        let mut spec = EvalSpec::default();
        spec.model = "model".into();
        spec.dataset = "data.jsonl".into();
        let result = try_parse("eval", spec.to_argv());
        assert!(
            result.is_ok(),
            "EvalSpec argv failed to parse: {}",
            result.unwrap_err()
        );
    }

    #[test]
    fn quantize_spec_round_trip() {
        let mut spec = QuantizeSpec::default();
        spec.model = "model".into();
        spec.output = "out.gguf".into();
        let result = try_parse("quantize", spec.to_argv());
        assert!(
            result.is_ok(),
            "QuantizeSpec argv failed to parse: {}",
            result.unwrap_err()
        );
    }

    // Commands::Merge requires `feature = "merge"` (a default feature but
    // not implied by trainer alone when building with --no-default-features).
    #[cfg(feature = "merge")]
    #[test]
    fn merge_spec_round_trip() {
        let mut spec = MergeSpec::default();
        spec.model_a = "a".into();
        spec.model_b = "b".into();
        spec.output = "out".into();
        let result = try_parse("merge", spec.to_argv());
        assert!(
            result.is_ok(),
            "MergeSpec argv failed to parse: {}",
            result.unwrap_err()
        );
    }

    #[test]
    fn fuse_spec_round_trip() {
        let mut spec = FuseSpec::default();
        spec.model = "model".into();
        spec.lora = "adapter".into();
        spec.output = "out".into();
        let result = try_parse("fuse", spec.to_argv());
        assert!(
            result.is_ok(),
            "FuseSpec argv failed to parse: {}",
            result.unwrap_err()
        );
    }

    #[test]
    fn dflash_spec_round_trip() {
        let mut spec = DflashSpec::default();
        spec.target = "target-model".into();
        spec.draft = "draft-model".into();
        spec.prompt = "Hello".into();
        let result = try_parse("dflash", spec.to_argv());
        assert!(
            result.is_ok(),
            "DflashSpec argv failed to parse: {}",
            result.unwrap_err()
        );
    }

    #[test]
    fn pack_experts_spec_round_trip() {
        let mut spec = PackExpertsSpec::default();
        spec.model = "model".into();
        let result = try_parse("pack-experts", spec.to_argv());
        assert!(
            result.is_ok(),
            "PackExpertsSpec argv failed to parse: {}",
            result.unwrap_err()
        );
    }

    #[test]
    fn tokenize_spec_round_trip() {
        let mut spec = TokenizeSpec::default();
        spec.input = "corpus.jsonl".into();
        spec.output = "shards/".into();
        spec.tokenizer = "Qwen/Qwen3-0.6B".into();
        let result = try_parse("tokenize", spec.to_argv());
        assert!(
            result.is_ok(),
            "TokenizeSpec argv failed to parse: {}",
            result.unwrap_err()
        );
    }

    // Smoke-test: default flags only contain the minimum required fields.
    // Any flag that is present in to_argv() but not recognized by clap would
    // have already failed the round-trip test above.
    #[test]
    fn train_spec_no_unexpected_flags() {
        let mut spec = TrainSpec::default();
        spec.model = "m".into();
        spec.dataset = "d".into();
        // Set every optional flag that the spec supports to ensure they all
        // appear in to_argv() and clap accepts them.
        spec.no_flash_attention = true;
        spec.no_sequence_packing = true;
        spec.no_jit_compilation = true;
        spec.no_metal_fused_optimizer = true;
        spec.cut_cross_entropy = true;
        spec.no_adaptive_lr = true;
        spec.ane = true;
        spec.resume = true;
        let result = try_parse("train", spec.to_argv());
        assert!(
            result.is_ok(),
            "TrainSpec flag-heavy argv failed to parse: {}",
            result.unwrap_err()
        );
    }
}
