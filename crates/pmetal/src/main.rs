//! PMetal CLI - LLM fine-tuning for Apple Silicon.

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
    /// NF4 (4-bit NormalFloat), the QLoRA paper's data type - recommended
    Nf4,
    /// NVFP4: E2M1 values with an E4M3 scale per 16 and an FP32 scale per tensor
    Fp4,
    /// 8-bit affine integers
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
    #[value(name = "dynamic")]
    Dynamic,
    /// 8-bit integer (near-lossless, ~1.06 bpw)
    #[value(name = "q8_0")]
    Q8_0,
    /// 8-bit integer with dot-product sum helper
    #[value(name = "q8_1")]
    Q8_1,
    /// 6-bit K-quant (high quality, ~0.80 bpw)
    #[value(name = "q6_k")]
    Q6K,
    /// 5-bit K-quant medium (good quality/size balance)
    #[value(name = "q5_k_m")]
    Q5KM,
    /// 5-bit K-quant small (slightly smaller than q5-k-m)
    #[value(name = "q5_k_s")]
    Q5KS,
    /// Legacy 5-bit symmetric GGML quantization
    #[value(name = "q5_0")]
    Q5_0,
    /// Legacy 5-bit affine GGML quantization
    #[value(name = "q5_1")]
    Q5_1,
    /// 4-bit K-quant medium (recommended 4-bit, ~0.58 bpw)
    #[value(name = "q4_k_m")]
    Q4KM,
    /// 4-bit K-quant small (smallest 4-bit K-quant)
    #[value(name = "q4_k_s")]
    Q4KS,
    /// Legacy 4-bit symmetric GGML quantization
    #[value(name = "q4_0")]
    Q4_0,
    /// Legacy 4-bit affine GGML quantization
    #[value(name = "q4_1")]
    Q4_1,
    /// 3-bit K-quant medium
    #[value(name = "q3_k_m")]
    Q3KM,
    /// 3-bit K-quant small
    #[value(name = "q3_k_s")]
    Q3KS,
    /// 3-bit K-quant large
    #[value(name = "q3_k_l")]
    Q3KL,
    /// 2-bit K-quant (lowest quality, ~0.37 bpw)
    #[value(name = "q2_k")]
    Q2K,
    /// 1-bit sign GGML quantization
    #[value(name = "q1_0")]
    Q1_0,
    /// Ternary 1.69-bit GGML quantization
    #[value(name = "tq1_0")]
    Tq1_0,
    /// Ternary 2.06-bit GGML quantization
    #[value(name = "tq2_0")]
    Tq2_0,
    /// MXFP4 block-floating GGML quantization
    #[value(name = "mxfp4")]
    Mxfp4,
    /// NVFP4 block-floating GGML quantization
    #[value(name = "nvfp4")]
    Nvfp4,
    /// BFloat16 (lossless-ish dense export, 2.0 bpw)
    #[value(name = "bf16")]
    Bf16,
    /// 16-bit float (lossless, 2.0 bpw)
    #[value(name = "f16")]
    F16,
    /// 32-bit float (lossless reference precision, 4.0 bpw)
    #[value(name = "f32")]
    F32,
}

impl QuantizeMethod {
    /// Canonical display name matching the GGUF K-quant naming convention.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Dynamic => "dynamic",
            Self::Q8_0 => "q8_0",
            Self::Q8_1 => "q8_1",
            Self::Q6K => "q6_k",
            Self::Q5KM => "q5_k_m",
            Self::Q5KS => "q5_k_s",
            Self::Q5_0 => "q5_0",
            Self::Q5_1 => "q5_1",
            Self::Q4KM => "q4_k_m",
            Self::Q4KS => "q4_k_s",
            Self::Q4_0 => "q4_0",
            Self::Q4_1 => "q4_1",
            Self::Q3KM => "q3_k_m",
            Self::Q3KS => "q3_k_s",
            Self::Q3KL => "q3_k_l",
            Self::Q2K => "q2_k",
            Self::Q1_0 => "q1_0",
            Self::Tq1_0 => "tq1_0",
            Self::Tq2_0 => "tq2_0",
            Self::Mxfp4 => "mxfp4",
            Self::Nvfp4 => "nvfp4",
            Self::Bf16 => "bf16",
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
            Self::Q8_1 => Some(GgmlType::Q8_1),
            Self::Q6K => Some(GgmlType::Q6K),
            // All Q5K size variants share the same K-quant block structure
            Self::Q5KM | Self::Q5KS => Some(GgmlType::Q5K),
            Self::Q5_0 => Some(GgmlType::Q5_0),
            Self::Q5_1 => Some(GgmlType::Q5_1),
            // All Q4K size variants share the same K-quant block structure
            Self::Q4KM | Self::Q4KS => Some(GgmlType::Q4K),
            Self::Q4_0 => Some(GgmlType::Q4_0),
            Self::Q4_1 => Some(GgmlType::Q4_1),
            // All Q3K size variants share the same K-quant block structure
            Self::Q3KM | Self::Q3KS | Self::Q3KL => Some(GgmlType::Q3K),
            Self::Q2K => Some(GgmlType::Q2K),
            Self::Q1_0 => Some(GgmlType::Q1_0),
            Self::Tq1_0 => Some(GgmlType::Tq1_0),
            Self::Tq2_0 => Some(GgmlType::Tq2_0),
            Self::Mxfp4 => Some(GgmlType::Mxfp4),
            Self::Nvfp4 => Some(GgmlType::Nvfp4),
            Self::Bf16 => Some(GgmlType::Bf16),
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

    /// Train a Gemma 4 assistant or Qwen3Next/Qwen3.6 MTP predictor
    #[command(name = "train-mtp")]
    #[cfg(feature = "trainer")]
    TrainMtp(crate::cli::train_mtp::TrainMtpArgs),

    /// Train a DFlash block-diffusion draft checkpoint
    #[command(name = "train-draft")]
    #[cfg(feature = "trainer")]
    TrainDraft(crate::cli::train_draft::TrainDraftArgs),

    /// LoRA fine-tune a DiffusionGemma block-diffusion model
    #[command(name = "train-diffusion")]
    #[cfg(feature = "trainer")]
    TrainDiffusion(crate::cli::train_diffusion::TrainDiffusionArgs),

    /// Pretrain a model from scratch (full-parameter, no LoRA)
    #[cfg(feature = "trainer")]
    Pretrain(crate::cli::pretrain::PretrainArgs),

    /// Run inference with a model
    Infer(crate::cli::infer::InferArgs),

    /// Answer typed questions about a state with a decision model (Clef)
    Decide(crate::cli::decide::DecideArgs),

    /// Download a model from HuggingFace
    ///
    /// Fetches the configs, tokenizer and one set of weights from a single
    /// commit, then checks them. Authenticates with `HF_TOKEN`, or the token
    /// stored by `hf auth login`. `HF_ENDPOINT` selects a mirror and
    /// `HF_HUB_OFFLINE=1` uses only the local cache.
    Download {
        /// Model ID, optionally with a GGUF quantization: `org/model-GGUF:Q4_K_M`
        model: String,

        /// Branch, tag or commit (default: the cached copy, else the default branch)
        #[arg(long)]
        revision: Option<String>,

        /// Weight format to fetch
        #[arg(long, value_enum, default_value_t = commands::download::WeightsArg::Auto)]
        format: commands::download::WeightsArg,

        /// GGUF quantization to fetch (e.g. Q4_K_M, Q8_0); implies --format gguf
        #[arg(long, value_name = "QUANT")]
        quant: Option<String>,

        /// Also hash large files against the Hub's SHA-256 digests (reads them back from disk)
        #[arg(long)]
        verify_sha256: bool,

        /// Show the commit, the files and their size, then stop
        #[arg(long)]
        dry_run: bool,

        /// Use only the local cache; never touch the network
        #[arg(long)]
        offline: bool,
    },

    /// Publish a model directory to HuggingFace (creates the repo if needed)
    ///
    /// Authenticates with `HF_TOKEN`, or the token stored by `hf auth login`.
    Upload {
        /// Local directory to publish (a fused model, an adapter, a quantized export)
        path: String,

        /// Target repository, as owner/name
        repo: String,

        /// Create the repository as private (no effect if it already exists)
        #[arg(long)]
        private: bool,

        /// Branch to commit to (default: the repository's default branch)
        #[arg(long)]
        revision: Option<String>,

        /// Commit message
        #[arg(short, long)]
        message: Option<String>,

        /// Open a pull request instead of committing to the branch
        #[arg(long)]
        create_pr: bool,

        /// Glob of files to leave out, relative to PATH (repeatable), e.g. "checkpoints/**"
        #[arg(long = "exclude", value_name = "GLOB")]
        exclude: Vec<String>,
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

        /// Only models with this tag (repeatable): gguf, mlx, moe, license:apache-2.0, ...
        #[arg(long = "tag", value_name = "TAG")]
        tags: Vec<String>,

        /// Only models from this user or organization
        #[arg(long)]
        author: Option<String>,

        /// Pipeline tag to search; `any` searches every kind of model
        #[arg(long, default_value = "text-generation")]
        pipeline_tag: String,

        /// Result order
        #[arg(long, value_enum, default_value_t = commands::search::SortArg::Downloads)]
        sort: commands::search::SortArg,
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
        #[arg(long, default_value = "Qwen/Qwen3.5-0.8B")]
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

    /// Preference optimization with LoRA: DPO, IPO, hinge, SimPO and ORPO on
    /// prompt/chosen/rejected pairs, or KTO on completions labelled good or bad.
    #[cfg(feature = "trainer")]
    #[command(visible_alias = "dpo")]
    Preference(crate::cli::preference::PreferenceArgs),

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

    /// Multi-machine cluster operations: discover peers, form a Thunderbolt-
    /// preferred ring, and run distributed training or inference.
    #[cfg(feature = "distributed")]
    Cluster(crate::cli::cluster::ClusterArgs),
}

impl Commands {
    /// Whether this command runs anything on MLX. The ones that don't skip the
    /// metallib search and the device check in `main`, so a missing or broken
    /// Metal library (or no network for the download fallback) can't stop
    /// them. A new command counts as using MLX until it is listed here.
    fn uses_mlx(&self) -> bool {
        match self {
            #[cfg(feature = "trainer")]
            Commands::Init { .. } => false,
            Commands::Download { .. }
            | Commands::Upload { .. }
            | Commands::Search { .. }
            | Commands::Dataset { .. }
            | Commands::Tokenize(_) => false,
            _ => true,
        }
    }
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

/// Modelfile export helpers for external Ollama registration.
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

    /// Generate a Modelfile for later registration with Ollama
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

        /// Output Modelfile path
        #[arg(short, long, default_value = "Modelfile")]
        output: String,

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

/// The gzip-compressed `mlx.metallib` built with the MLX this binary links,
/// embedded by build.rs. Empty when the build had none.
#[cfg(target_os = "macos")]
static MLX_METALLIB_GZ: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mlx.metallib.gz"));

/// SHA-256 of the embedded metallib, uncompressed.
const EMBEDDED_METALLIB_SHA256: Option<&str> = option_env!("PMETAL_EMBEDDED_METALLIB_SHA256");

/// Write the embedded metallib into the metallib cache and return its path:
/// the user's (`~/.cache/pmetal/lib`), or one under the temporary directory
/// when that can't be written.
#[cfg(target_os = "macos")]
fn extract_embedded_metallib(sha256: &str) -> Option<PathBuf> {
    use flate2::read::GzDecoder;
    use pmetal_runtime::metallib::Cache;
    use std::io::Read;

    if MLX_METALLIB_GZ.is_empty() {
        return None;
    }
    let mut bytes = Vec::new();
    if let Err(e) = GzDecoder::new(MLX_METALLIB_GZ).read_to_end(&mut bytes) {
        tracing::warn!("the embedded mlx.metallib does not decompress: {e}");
        return None;
    }
    let caches = Cache::user()
        .into_iter()
        .chain([Cache::at(std::env::temp_dir().join("pmetal-metallib"))]);
    for cache in caches {
        match cache.install(&bytes, sha256) {
            Ok(path) => {
                tracing::info!("Extracted the embedded mlx.metallib to {}", path.display());
                return Some(path);
            }
            Err(e) => tracing::warn!(
                dir = %cache.dir().display(),
                "could not write the embedded mlx.metallib: {e}"
            ),
        }
    }
    None
}

#[cfg(not(target_os = "macos"))]
fn extract_embedded_metallib(_sha256: &str) -> Option<PathBuf> {
    None
}

/// Point MLX at the `mlx.metallib` built with the MLX this binary links,
/// before anything builds its Metal device.
///
/// In order (`pmetal_runtime::metallib::locate`): `PMETAL_METALLIB_PATH`; the
/// metallib cache's entry for the embedded copy; the file the build produced,
/// while it exists; an `mlx.metallib` next to the executable, as the release
/// tarball ships it. Failing those, the embedded copy is written into the
/// cache. Each binary's copy is keyed by MLX revision and content hash, so
/// installed pmetal versions don't overwrite each other's.
fn ensure_metallib() {
    use pmetal_runtime::metallib;

    let colocated: Vec<PathBuf> = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|dir| dir.join("mlx.metallib")))
        .into_iter()
        .collect();
    let sha256 = EMBEDDED_METALLIB_SHA256.or(metallib::BUILT_SHA256);
    let found = metallib::locate(sha256, &colocated, |path, e| {
        tracing::warn!(path = %path.display(), "skipping unusable mlx.metallib: {e}");
    })
    .map(|located| located.path)
    .or_else(|| EMBEDDED_METALLIB_SHA256.and_then(extract_embedded_metallib));

    match found {
        Some(path) => {
            tracing::debug!(path = %path.display(), "Using mlx.metallib");
            pmetal_runtime::inline_array::set_metallib_path(&path.to_string_lossy());
        }
        None => eprintln!(
            "\x1b[1;33mwarning:\x1b[0m no mlx.metallib for this build's MLX ({revision}) was \
             found, and none is embedded in this binary.\n\
             Set PMETAL_METALLIB_PATH to the mlx.metallib built with it, or reinstall pmetal.",
            revision = metallib::MLX_REVISION,
        ),
    }
}

/// Synchronous entry point.
///
/// `ensure_metallib` runs here rather than in `tokio_main` because its
/// download fallback blocks on a runtime of its own, which can't start inside
/// `#[tokio::main]`. Logging is set up first so its messages are recorded, and
/// argument parsing before that, so `--help` and usage errors exit without
/// searching for (or downloading) a metallib. MLX's Metal device is then built
/// before any command that uses MLX, which measured no slower than not doing
/// it; the others (`Commands::uses_mlx`) skip both steps.
fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // The TUI owns the terminal, so it logs to the file only.
    #[cfg(feature = "dashboard")]
    let is_tui = matches!(cli.command, Commands::Tui { .. });
    #[cfg(not(feature = "dashboard"))]
    let is_tui = false;
    init_logging(if is_tui { "tui" } else { "cli" }, is_tui);

    if cli.command.uses_mlx() {
        ensure_metallib();
        // Load the library now: if MLX can't, the command would otherwise fail
        // later, possibly by aborting from a runtime call with no error guard.
        pmetal_runtime::inline_array::init_device().map_err(|e| {
            anyhow::anyhow!(
                "MLX could not load its Metal library: {e}\n\
                 Remove that file, or set PMETAL_METALLIB_PATH to a working mlx.metallib."
            )
        })?;
        #[cfg(feature = "metal")]
        let _ = pmetal_metal::context::MetalContext::device_available();
    }
    tokio_main(cli)
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

    // hf-hub's Xet transfer client logs every request at INFO, several lines
    // per downloaded file. Hold it (`xet_*` and `hf_xet`) to warnings; a
    // more specific `RUST_LOG=xet_client=info` still wins.
    let filter = tracing_subscriber::EnvFilter::from_default_env()
        .add_directive(tracing::Level::INFO.into())
        .add_directive("xet=warn".parse().expect("static directive"))
        .add_directive("hf_xet=warn".parse().expect("static directive"));

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
async fn tokio_main(cli: Cli) -> anyhow::Result<()> {
    // Capture global flags before `cli` is moved into the `match` below.
    #[cfg(feature = "trainer")]
    let log_events_path = cli.log_events.clone();

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
                dora,
                rslora,
                lora_dropout,
                learning_rate,
                batch_size,
                epochs,
                max_seq_len,
                gradient_accumulation_steps,
                max_grad_norm,
                resume,
                quantization,
                quant_block_size,
                double_quant,
                no_sequence_packing,
                pack_max_seq_len,
                no_gradient_checkpointing,
                gradient_checkpointing_layers,
                log_metrics,
                embedding_lr,
                label_smoothing,
                z_loss,
                neftune_alpha,
                loss_scale,
                warmup_steps,
                lr_schedule,
                weight_decay,
                optimizer,
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
            let lr_scheduler = lr_schedule
                .parse::<pmetal_core::LrSchedulerType>()
                .map_err(|e| anyhow::anyhow!("--lr-schedule: {e}"))?;

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
                    dropout: lora_dropout,
                    use_rslora: rslora,
                    use_dora: dora,
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
                    optimizer,
                    embedding_learning_rate: embedding_lr.map(|v| v as f64),
                    label_smoothing,
                    z_loss,
                    neftune_noise_alpha: neftune_alpha,
                    ..Default::default()
                },
                columns,
                dispatch: orchestrator::DispatchConfig {
                    sequence_packing: !no_sequence_packing,
                    pack_max_seq_len,
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
            let mut extra_callbacks: Vec<Box<dyn pmetal_core::TrainingCallback>> = Vec::new();
            if let Some(path) = log_events_path {
                // Created before training starts, so a bad path fails at once
                // rather than partway through.
                let file = std::fs::File::create(&path).map_err(|e| {
                    anyhow::anyhow!("--log-events: could not create '{}': {e}", path.display())
                })?;
                extra_callbacks.push(Box::new(pmetal_core::TrainingCallbackToSink::new(
                    job_config.model_id.clone(),
                    pmetal_core::JsonlSink::new(file),
                )));
            }

            orchestrator::run_training(job_config, None, extra_callbacks).await?;
        }

        #[cfg(feature = "trainer")]
        Commands::TrainMtp(args) => {
            commands::train_mtp::run_train_mtp(args).await?;
        }

        #[cfg(feature = "trainer")]
        Commands::TrainDraft(args) => {
            commands::train_draft::run_train_draft(args).await?;
        }

        #[cfg(feature = "trainer")]
        Commands::TrainDiffusion(args) => {
            commands::train_diffusion::run_train_diffusion(args).await?;
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
                api_key,
                cors_origin,
                request_timeout,
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
                draft_model,
                continuous_batch,
                cb_max_slots,
                cb_max_queue_depth,
                cb_block_size,
                cb_max_blocks,
                cb_exact,
            } = args;
            #[cfg(feature = "ane")]
            let ane_enabled = ane;
            #[cfg(not(feature = "ane"))]
            let ane_enabled = false;

            #[cfg(feature = "ane")]
            let serve_ane_max_seq_len = ane_max_seq_len;
            #[cfg(not(feature = "ane"))]
            let serve_ane_max_seq_len = 4096;
            #[cfg(feature = "ane")]
            let serve_draft_model = draft_model;
            #[cfg(not(feature = "ane"))]
            let serve_draft_model = None;

            commands::serve::run_serve(
                model,
                port,
                host,
                commands::serve::Access {
                    api_keys: api_key,
                    cors_origins: cors_origin,
                    request_timeout_secs: request_timeout,
                },
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
                serve_draft_model,
                continuous_batch,
                cb_max_slots,
                cb_max_queue_depth,
                cb_block_size,
                cb_max_blocks,
                cb_exact,
            )
            .await?;
        }

        Commands::Decide(args) => {
            let crate::cli::decide::DecideArgs {
                model,
                request,
                max_length,
                compact,
            } = args;
            commands::decide::run_decide(&model, &request, max_length, compact).await?;
        }

        Commands::Infer(args) => {
            let crate::cli::infer::InferArgs {
                model,
                lora,
                prompt,
                image,
                video,
                video_fps,
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
                reasoning_effort,
                no_preserve_thinking,
                chat_template_kwargs,
                mode,
                backend,
                draft_model,
                mtp,
                mtp_model,
                mtp_draft_tokens,
                hide_thinking,
                tools,
                fp8,
                experts_dir,
                #[cfg(feature = "ane")]
                ane,
                #[cfg(feature = "ane")]
                ane_max_seq_len,
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
            let template_kwargs = pmetal_data::chat_templates::ChatTemplateKwargs::from_controls(
                chat_template_kwargs.as_deref(),
                no_thinking,
                reasoning_effort.as_deref(),
                no_preserve_thinking,
            )
            .map_err(anyhow::Error::msg)?;

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
                template_kwargs,
                mode,
                backend,
                draft_model.as_deref(),
                mtp,
                mtp_model.as_deref(),
                mtp_draft_tokens,
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
                &image,
                &video,
                video_fps,
            )
            .await?;
        }

        Commands::Download {
            model,
            revision,
            format,
            quant,
            verify_sha256,
            dry_run,
            offline,
        } => {
            tracing::info!(model = %model, "Downloading model");
            commands::download::run_download(commands::download::DownloadRequest {
                model,
                revision,
                format,
                quant,
                verify_sha256,
                dry_run,
                offline,
            })
            .await?;
        }

        Commands::Upload {
            path,
            repo,
            private,
            revision,
            message,
            create_pr,
            exclude,
        } => {
            tracing::info!(path = %path, repo = %repo, "Uploading model");
            let (bar, progress) = commands::download::transfer_bar()?;
            let options = pmetal_hub::UploadOptions {
                private,
                revision,
                commit_message: message,
                create_pr,
                ignore: exclude,
                progress: Some(progress),
                ..Default::default()
            };
            let report = pmetal_hub::upload_model(&path, &repo, None, &options).await;
            bar.finish_and_clear();
            let report = report?;
            println!("Uploaded {path} to {}", report.repo_url);
            if let Some(commit) = report.commit_url {
                println!("Commit: {commit}");
            }
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
            tags,
            author,
            pipeline_tag,
            sort,
        } => {
            commands::search::run_search(commands::search::SearchRequest {
                query,
                limit,
                download,
                detailed,
                json,
                tags,
                author,
                pipeline_tag,
                sort,
            })
            .await?;
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
                compare_greedy,
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
                compare_greedy,
            )
            .await?;
        }

        Commands::Memory { json } => {
            let stats = pmetal_runtime::memory::get_memory_stats();
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
                eval_dataset,
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
                on_policy,
                on_policy_max_tokens,
                lora_r,
                lora_alpha,
                learning_rate,
                optimizer,
                batch_size,
                epochs,
                max_seq_len,
                seed,
                checkpoint_every,
                resume,
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
                eval_dataset,
                &output,
                &method,
                &loss_type,
                temperature,
                alpha,
                rationale,
                rationale_weight,
                on_policy,
                on_policy_max_tokens,
                lora_r,
                lora_alpha,
                learning_rate,
                optimizer,
                batch_size,
                epochs,
                max_seq_len,
                seed,
                checkpoint_every,
                resume,
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
                optimizer,
                epochs,
                lora_r,
                lora_alpha,
                max_seq_len,
                max_completion_length,
                seed,
                checkpoint_every,
                resume,
                dapo,
                loss_type,
                num_iterations,
                reasoning_rewards,
                vlm,
                max_image_size,
                reward_model,
                reward_model_max_length,
                reward_model_weight,
                reward_model_template,
                async_rewards,
                grpo_kv_bits,
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
                optimizer,
                epochs,
                lora_r,
                lora_alpha,
                max_seq_len,
                max_completion_length,
                seed,
                checkpoint_every,
                resume,
                dapo,
                &loss_type,
                num_iterations,
                reasoning_rewards,
                vlm,
                max_image_size,
                reward_model,
                reward_model_max_length,
                reward_model_weight,
                reward_model_template,
                async_rewards,
                grpo_kv_bits,
                log_metrics,
                true,
                Vec::new(),
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
                log_metrics,
                true,
                Vec::new(),
                prompt_column,
                response_column,
            )
            .await?;
        }

        #[cfg(feature = "trainer")]
        Commands::Preference(args) => {
            let mut spec = pmetal_core::jobs::PreferenceSpec::from(args);
            spec.normalize().map_err(|errs| {
                anyhow::anyhow!(
                    "{}",
                    errs.iter()
                        .map(|e| e.message.as_str())
                        .collect::<Vec<_>>()
                        .join("; ")
                )
            })?;
            commands::preference::run_preference(spec, true, Vec::new()).await?;
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
                lora,
                lora_r,
                lora_alpha,
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
                lora.then_some((lora_r, lora_alpha)),
            )
            .await?;
        }

        #[cfg(feature = "trainer")]
        Commands::Pretrain(args) => {
            let crate::cli::pretrain::PretrainArgs {
                arch,
                shards,
                eval_shards,
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
                document_mask,
                gradient_accumulation_steps,
                log_every,
                eval_every,
                eval_batches,
                seed,
            } = args;
            use pmetal_data::streaming::{StreamConfig, StreamingShardReader};
            use pmetal_runtime::compat::random;
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
            let lr_sched = lr_schedule
                .parse::<pmetal_core::LrSchedulerType>()
                .map_err(|e| anyhow::anyhow!("--lr-schedule: {e}"))?;

            // Build model via factory (supports llama, qwen2, qwen3, gemma, gemma4, mistral, phi, gpt-oss)
            use pmetal_runtime::compat::module::ModuleParameters;
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

            let mut optimizer =
                pmetal_runtime::compat::optimizers::AdamWBuilder::new(learning_rate)
                    .weight_decay(weight_decay)
                    .betas((0.9, 0.95))
                    .eps(1e-8)
                    .build()?;
            let mut start_step = 0usize;
            let mut resume_position = None;

            // Resume from checkpoint if requested
            if let Some(ref ckpt_dir) = resume {
                let meta = pretrain::load_checkpoint(
                    std::path::Path::new(ckpt_dir),
                    &mut model,
                    &mut optimizer,
                )?;
                start_step = meta.step as usize;
                resume_position = meta.stream_position;
                println!("Resumed from step {}, loss {:.4}", meta.step, meta.loss);
            }

            // Streaming data pipeline
            let stream_config = StreamConfig {
                shard_paths,
                seq_len,
                batch_size,
                eos_token_id,
                resume_from: resume_position,
            };
            let mut reader = StreamingShardReader::new(stream_config)?;
            if resume.is_some() && resume_position.is_none() && start_step > 0 {
                let batches_to_skip = start_step.saturating_mul(gradient_accumulation_steps.max(1));
                for _ in 0..batches_to_skip {
                    reader.try_next_batch()?.ok_or_else(|| {
                        anyhow::anyhow!("stream exhausted while seeking resume step")
                    })?;
                }
            }

            // Convert streaming batches to MLX arrays. A shard that cannot be
            // read is an error that stops the run, not the end of the data.
            let batch_iter = std::iter::from_fn(move || {
                let (rows, pos) = match reader.try_next_rows() {
                    Ok(Some(batch)) => batch,
                    Ok(None) => return None,
                    Err(e) => return Some(Err(pretrain::PretrainError::Data(e.to_string()))),
                };
                let flat: Vec<i32> = rows
                    .iter()
                    .flat_map(|row| row.tokens.iter().map(|&t| t as i32))
                    .collect();
                let tokens = pmetal_runtime::InlineArray::from_i32_slice_shaped(
                    &flat,
                    &[rows.len() as i32, seq_len as i32],
                );
                Some(Ok(pretrain::PretrainBatch {
                    tokens,
                    position: Some(pos),
                    doc_lengths: document_mask
                        .then(|| rows.into_iter().map(|row| row.doc_lengths).collect()),
                }))
            });

            let eval_iter = if !eval_shards.is_empty() && eval_every > 0 {
                let eval_paths: Vec<std::path::PathBuf> =
                    eval_shards.iter().map(std::path::PathBuf::from).collect();
                let eval_reader = StreamingShardReader::new(StreamConfig {
                    shard_paths: eval_paths,
                    seq_len,
                    batch_size,
                    eos_token_id,
                    resume_from: None,
                })?;
                Some(eval_reader.map(move |(batch, _pos)| {
                    let flat: Vec<i32> = batch
                        .iter()
                        .flat_map(|seq| seq.iter().map(|&t| t as i32))
                        .collect();
                    pmetal_runtime::InlineArray::from_i32_slice_shaped(
                        &flat,
                        &[batch.len() as i32, seq_len as i32],
                    )
                }))
            } else {
                None
            };

            // Run
            let losses = pretrain::run_pretrain_with_state(
                &mut model, &config, batch_iter, optimizer, start_step, eval_iter,
            )?;

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

        #[cfg(feature = "distributed")]
        Commands::Cluster(args) => {
            commands::cluster::run(args.command).await?;
        }
    }

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
//   cargo test -p pmetal --features trainer -- argv_roundtrip
// ---------------------------------------------------------------------------
#[cfg(all(test, feature = "trainer", feature = "core"))]
mod argv_roundtrip {
    use super::{Cli, Commands};
    use clap::Parser;
    use pmetal_core::JobFields;
    #[cfg(feature = "serve")]
    use pmetal_core::jobs::ServeSpec;
    use pmetal_core::jobs::{
        BenchSpec, DecideSpec, DflashSpec, DistillSpec, EmbedTrainSpec, EvalSpec, FuseSpec,
        GrpoSpec, InferSpec, MergeSpec, PackExpertsSpec, PretrainSpec, QuantizeSpec, RlkdSpec,
        TokenizeSpec, TrainSpec,
    };

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
        let spec = TrainSpec {
            model: "Qwen/Qwen3-0.6B".into(),
            dataset: "data/train.jsonl".into(),
            ..Default::default()
        };
        let result = try_parse("train", spec.to_argv());
        assert!(
            result.is_ok(),
            "TrainSpec argv failed to parse: {}",
            result.unwrap_err()
        );
    }

    #[cfg(feature = "trainer")]
    #[test]
    fn train_mtp_draft_and_diffusion_specs_round_trip() {
        use pmetal_core::jobs::{TrainDiffusionSpec, TrainDraftSpec, TrainMtpSpec};
        for family in ["auto", "qwen3-next", "gemma4"] {
            let spec = TrainMtpSpec {
                model: "m".into(),
                family: family.into(),
                shards: vec!["a.bin".into(), "b.bin".into()],
                lr_schedule: "wsd".into(),
                init_mtp: Some("mtp".into()),
                ..Default::default()
            };
            let parsed = try_parse("train-mtp", spec.to_argv())
                .unwrap_or_else(|e| panic!("family {family}: {e}"));
            let Commands::TrainMtp(args) = parsed else {
                panic!("not train-mtp")
            };
            assert_eq!(args.shards, ["a.bin", "b.bin"]);
            assert_eq!(args.lr_schedule, pmetal_core::LrSchedulerType::Wsd);
        }
        let spec = TrainDraftSpec {
            target: "t".into(),
            draft_config: Some("c.json".into()),
            shards: vec!["a.bin".into()],
            ..Default::default()
        };
        let Commands::TrainDraft(args) = try_parse("train-draft", spec.to_argv()).unwrap() else {
            panic!("not train-draft")
        };
        assert_eq!(args.draft_config.as_deref(), Some("c.json"));
        for quantization in ["none", "nf4", "fp4", "int8"] {
            let spec = TrainDiffusionSpec {
                model: "m".into(),
                dataset: "d".into(),
                quantization: quantization.into(),
                double_quant: quantization == "nf4",
                ..Default::default()
            };
            let parsed = try_parse("train-diffusion", spec.to_argv())
                .unwrap_or_else(|e| panic!("quantization {quantization}: {e}"));
            let Commands::TrainDiffusion(args) = parsed else {
                panic!("not train-diffusion")
            };
            assert!(args.lora_targets.is_empty());
        }
    }

    #[cfg(feature = "trainer")]
    #[test]
    fn train_diffusion_corrupted_only_can_be_turned_off() {
        // `--corrupted-only` defaults to true and used to be a set-true flag,
        // so the documented "when false, average over the whole canvas" mode
        // could not be selected.
        let parse = |extra: &[&str]| {
            let mut argv: Vec<String> = ["--model", "m", "--dataset", "d"]
                .map(String::from)
                .to_vec();
            argv.extend(extra.iter().map(|s| s.to_string()));
            match try_parse("train-diffusion", argv) {
                Ok(Commands::TrainDiffusion(args)) => args.corrupted_only,
                Ok(_) => panic!("not train-diffusion"),
                Err(e) => panic!("{e}"),
            }
        };
        assert!(parse(&[]));
        assert!(parse(&["--corrupted-only", "true"]));
        assert!(!parse(&["--corrupted-only", "false"]));
    }

    #[cfg(feature = "trainer")]
    #[test]
    fn a_schedule_the_trainer_supports_reaches_it_and_a_typo_is_refused() {
        // `train-mtp` and `train-draft` used to read any name but constant
        // and linear as cosine, so `wsd` trained on the wrong curve.
        for (subcommand, model_flag) in [("train-mtp", "--model"), ("train-draft", "--target")] {
            let parse = |schedule: &str| {
                try_parse(
                    subcommand,
                    [model_flag, "t", "--shards", "s", "--lr-schedule", schedule]
                        .map(String::from)
                        .to_vec(),
                )
            };
            let parsed = parse("wsd").unwrap_or_else(|e| panic!("{subcommand}: {e}"));
            let got = match parsed {
                Commands::TrainMtp(a) => a.lr_schedule,
                Commands::TrainDraft(a) => a.lr_schedule,
                _ => panic!("{subcommand} parsed as another command"),
            };
            assert_eq!(got, pmetal_core::LrSchedulerType::Wsd);
            let err = parse("cosin").unwrap_err();
            assert!(err.contains("cosin"), "{subcommand}: {err}");
        }
    }

    #[test]
    fn distill_spec_round_trip() {
        let spec = DistillSpec {
            teacher: "Qwen/Qwen3-7B".into(),
            student: "Qwen/Qwen3-0.6B".into(),
            dataset: "data.jsonl".into(),
            ..Default::default()
        };
        let result = try_parse("distill", spec.to_argv());
        assert!(
            result.is_ok(),
            "DistillSpec argv failed to parse: {}",
            result.unwrap_err()
        );
    }

    #[test]
    fn grpo_spec_round_trip() {
        let spec = GrpoSpec {
            model: "model".into(),
            dataset: "data.jsonl".into(),
            ..Default::default()
        };
        let result = try_parse("grpo", spec.to_argv());
        assert!(
            result.is_ok(),
            "GrpoSpec argv failed to parse: {}",
            result.unwrap_err()
        );
    }

    /// Every field survives spec → argv → clap → spec, defaults included,
    /// so the spec's `default_*` attributes and the CLI's defaults agree.
    #[test]
    fn preference_spec_round_trip() {
        use pmetal_core::jobs::PreferenceSpec;
        let defaults = PreferenceSpec {
            model: "model".into(),
            dataset: "data.jsonl".into(),
            ..Default::default()
        };
        let custom = PreferenceSpec {
            loss: "kto".into(),
            beta: Some(0.3),
            max_steps: Some(12),
            undesirable_weight: 1.5,
            log_metrics: Some("m.jsonl".into()),
            ..defaults.clone()
        };
        for spec in [defaults, custom] {
            let parsed = match try_parse("preference", spec.to_argv()) {
                Ok(Commands::Preference(args)) => PreferenceSpec::from(args),
                Ok(_) => panic!("parsed as another command"),
                Err(e) => panic!("PreferenceSpec argv failed to parse: {e}"),
            };
            assert_eq!(
                serde_json::to_value(&parsed).unwrap(),
                serde_json::to_value(&spec).unwrap()
            );
        }
        assert!(matches!(
            try_parse(
                "dpo",
                vec!["-m".into(), "x".into(), "-d".into(), "y".into()]
            ),
            Ok(Commands::Preference(_))
        ));
    }

    #[test]
    fn rlkd_spec_round_trip() {
        let spec = RlkdSpec {
            model: "model".into(),
            teacher_model: "teacher".into(),
            dataset: "data.jsonl".into(),
            ..Default::default()
        };
        let result = try_parse("rlkd", spec.to_argv());
        assert!(
            result.is_ok(),
            "RlkdSpec argv failed to parse: {}",
            result.unwrap_err()
        );
    }

    #[test]
    fn embed_train_spec_round_trip() {
        let spec = EmbedTrainSpec {
            model: "model".into(),
            dataset: "data.jsonl".into(),
            ..Default::default()
        };
        let result = try_parse("embed-train", spec.to_argv());
        assert!(
            result.is_ok(),
            "EmbedTrainSpec argv failed to parse: {}",
            result.unwrap_err()
        );
    }

    #[test]
    fn pretrain_spec_round_trip() {
        let spec = PretrainSpec {
            arch: "gpt-oss".into(),
            ..Default::default()
        };
        let result = try_parse("pretrain", spec.to_argv());
        assert!(
            result.is_ok(),
            "PretrainSpec argv failed to parse: {}",
            result.unwrap_err()
        );
    }

    #[test]
    fn infer_spec_round_trip() {
        let spec = InferSpec {
            model: "model".into(),
            prompt: "Hello".into(),
            ..Default::default()
        };
        let result = try_parse("infer", spec.to_argv());
        assert!(
            result.is_ok(),
            "InferSpec argv failed to parse: {}",
            result.unwrap_err()
        );
    }

    #[test]
    fn infer_video_flags_parse() {
        let parse = |extra: &[&str]| {
            let mut argv = vec![
                "--model".to_string(),
                "m".into(),
                "--prompt".into(),
                "p".into(),
            ];
            argv.extend(extra.iter().map(|s| s.to_string()));
            try_parse("infer", argv)
        };
        match parse(&["--video", "a", "--video", "b", "--video-fps", "2.5"]) {
            Ok(Commands::Infer(args)) => {
                assert_eq!(args.video, ["a", "b"]);
                assert_eq!(args.video_fps, Some(2.5));
            }
            other => panic!("{other:?}"),
        }
        assert!(parse(&["--video-fps", "2"]).is_err(), "fps without a video");
        for fps in ["0", "-1", "inf", "NaN", "two"] {
            assert!(
                parse(&["--video", "a", "--video-fps", fps]).is_err(),
                "{fps}"
            );
        }
    }

    /// `--backend` is the one way to pick a generation path: the spec, the
    /// help text and the parser offer the same variants, and the per-path
    /// flags it replaced are gone.
    #[test]
    fn infer_backend_is_the_only_path_selector() {
        use clap::CommandFactory;
        use pmetal_data::inference_config::InferenceBackend;

        let names: Vec<&str> = InferenceBackend::ALL.iter().map(|b| b.as_str()).collect();
        let spec_options = InferSpec::field_descriptors()
            .iter()
            .find(|d| d.name == "backend")
            .map(|d| match d.kind {
                pmetal_core::FieldKind::Enum { options } => options.to_vec(),
                other => panic!("backend is not an enum field: {other:?}"),
            })
            .expect("InferSpec has a backend field");
        assert_eq!(spec_options, names);

        let cli = Cli::command();
        let backend = cli
            .find_subcommand("infer")
            .and_then(|infer| infer.get_arguments().find(|a| a.get_id() == "backend"))
            .expect("infer has --backend");
        let help = backend.get_help().expect("--backend has help").to_string();
        let listed = help
            .split_once(':')
            .map(|(_, list)| list.trim().trim_end_matches('.'))
            .expect("--backend help lists the variants");
        assert_eq!(listed.split(" | ").collect::<Vec<_>>(), names);

        for name in &names {
            let spec = InferSpec {
                model: "model".into(),
                prompt: "Hello".into(),
                backend: (*name).to_string(),
                ..Default::default()
            };
            assert!(try_parse("infer", spec.to_argv()).is_ok(), "{name}");
        }
        for legacy in ["--compiled", "--metal-sampler", "--minimal"] {
            let argv = vec![
                "--model".to_string(),
                "m".into(),
                "--prompt".into(),
                "p".into(),
                legacy.into(),
            ];
            assert!(try_parse("infer", argv).is_err(), "{legacy} still parses");
        }
    }

    // Commands::Serve is only compiled when `--features serve` is active.
    #[cfg(feature = "serve")]
    #[test]
    fn serve_spec_round_trip() {
        let spec = ServeSpec {
            model: "model".into(),
            ..Default::default()
        };
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
        let spec = EvalSpec {
            model: "model".into(),
            dataset: "data.jsonl".into(),
            ..Default::default()
        };
        let result = try_parse("eval", spec.to_argv());
        assert!(
            result.is_ok(),
            "EvalSpec argv failed to parse: {}",
            result.unwrap_err()
        );
    }

    #[test]
    fn quantize_spec_round_trip() {
        let spec = QuantizeSpec {
            model: "model".into(),
            output: "out.gguf".into(),
            ..Default::default()
        };
        let result = try_parse("quantize", spec.to_argv());
        assert!(
            result.is_ok(),
            "QuantizeSpec argv failed to parse: {}",
            result.unwrap_err()
        );
    }

    #[test]
    fn quantize_methods_parse_canonical_names() {
        let methods = [
            "dynamic", "q1_0", "q2_k", "q3_k_s", "q3_k_m", "q3_k_l", "q4_0", "q4_1", "q4_k_s",
            "q4_k_m", "q5_0", "q5_1", "q5_k_s", "q5_k_m", "q6_k", "q8_0", "q8_1", "tq1_0", "tq2_0",
            "mxfp4", "nvfp4", "bf16", "f16", "f32",
        ];

        for method in methods {
            let argv = [
                "pmetal", "quantize", "--model", "model", "--output", "out.gguf", "--method",
                method,
            ];
            let result = Cli::try_parse_from(argv);
            if let Err(err) = result {
                panic!("quantize method {method} failed to parse: {err}");
            }
        }
    }

    // Commands::Merge requires `feature = "merge"` (a default feature but
    // not implied by trainer alone when building with --no-default-features).
    #[cfg(feature = "merge")]
    #[test]
    fn merge_spec_round_trip() {
        let spec = MergeSpec {
            model_a: "a".into(),
            model_b: "b".into(),
            output: "out".into(),
            ..Default::default()
        };
        let result = try_parse("merge", spec.to_argv());
        assert!(
            result.is_ok(),
            "MergeSpec argv failed to parse: {}",
            result.unwrap_err()
        );
    }

    #[test]
    fn fuse_spec_round_trip() {
        let spec = FuseSpec {
            model: "model".into(),
            lora: "adapter".into(),
            output: "out".into(),
            ..Default::default()
        };
        let result = try_parse("fuse", spec.to_argv());
        assert!(
            result.is_ok(),
            "FuseSpec argv failed to parse: {}",
            result.unwrap_err()
        );
    }

    #[test]
    fn dflash_spec_round_trip() {
        let spec = DflashSpec {
            target: "target-model".into(),
            draft: "draft-model".into(),
            prompt: "Hello".into(),
            ..Default::default()
        };
        let result = try_parse("dflash", spec.to_argv());
        assert!(
            result.is_ok(),
            "DflashSpec argv failed to parse: {}",
            result.unwrap_err()
        );
    }

    #[test]
    fn pack_experts_spec_round_trip() {
        let spec = PackExpertsSpec {
            model: "model".into(),
            ..Default::default()
        };
        let result = try_parse("pack-experts", spec.to_argv());
        assert!(
            result.is_ok(),
            "PackExpertsSpec argv failed to parse: {}",
            result.unwrap_err()
        );
    }

    #[test]
    fn decide_spec_round_trip() {
        let spec = DecideSpec {
            model: "Cloudflare/clef-flash".into(),
            request: "request.json".into(),
            compact: true,
            ..Default::default()
        };
        let result = try_parse("decide", spec.to_argv());
        assert!(
            result.is_ok(),
            "DecideSpec argv failed to parse: {}",
            result.unwrap_err()
        );
    }

    #[test]
    fn tokenize_spec_round_trip() {
        let spec = TokenizeSpec {
            input: "corpus.jsonl".into(),
            output: "shards/".into(),
            tokenizer: "Qwen/Qwen3-0.6B".into(),
            ..Default::default()
        };
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
        let spec = TrainSpec {
            model: "m".into(),
            dataset: "d".into(),
            no_sequence_packing: true,
            cut_cross_entropy: true,
            no_adaptive_lr: true,
            ane: true,
            resume: true,
            ..Default::default()
        };
        let result = try_parse("train", spec.to_argv());
        assert!(
            result.is_ok(),
            "TrainSpec flag-heavy argv failed to parse: {}",
            result.unwrap_err()
        );
    }
}

/// The command lines the MCP server builds are ones this CLI parses.
///
/// The MCP tools for the dataset and Ollama commands spell their flags in a
/// table, and the tools for everything else add a few of their own; the parser
/// here is the judge of whether those flags exist.
#[cfg(all(test, feature = "mcp", feature = "trainer"))]
mod mcp_tools_parse {
    use super::Cli;
    use clap::Parser;
    use pmetal_mcp::policy::PathPolicy;
    use serde_json::{Value, json};

    fn parses(label: &str, argv: &[String]) {
        let mut full = vec!["pmetal".to_string()];
        full.extend(argv.iter().cloned());
        if let Err(e) = Cli::try_parse_from(&full) {
            panic!(
                "{label}: `pmetal {}` is not a command line the CLI accepts:\n{e}",
                argv.join(" ")
            );
        }
    }

    #[test]
    fn every_dataset_and_ollama_tool_builds_a_command_the_cli_parses() {
        let root = std::env::temp_dir().canonicalize().unwrap();
        let mut checked = 0;
        for tool in pmetal_mcp::cli_tools::CLI_TOOLS {
            for argv in tool.example_invocations(&root) {
                parses(tool.name, &argv);
                checked += 1;
            }
        }
        assert!(checked >= 28, "only {checked} command lines were checked");
    }

    #[test]
    fn every_spec_tool_builds_a_command_the_cli_parses() {
        let root = std::env::temp_dir().canonicalize().unwrap();
        let policy = PathPolicy::new(root.clone(), vec![root]);
        for tool in pmetal_mcp::spec_tool_list() {
            // `pmetal serve` exists only in a build with the `serve` feature.
            if cfg!(not(feature = "serve")) && tool.name() == "start_serve" {
                continue;
            }
            let definition = serde_json::to_value(tool.definition()).unwrap();
            // Each required parameter is a Hub-style name, which is also a valid
            // relative path for the ones that name files.
            let args: serde_json::Map<String, Value> = definition["inputSchema"]["required"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|name| {
                    let name = name.as_str().unwrap();
                    let list = definition["inputSchema"]["properties"][name]["type"] == "array";
                    let value = if list {
                        json!(["owner/name"])
                    } else {
                        json!("owner/name")
                    };
                    (name.to_string(), value)
                })
                .collect();
            let prepared = tool
                .prepare(Value::Object(args), &policy)
                .unwrap_or_else(|e| panic!("{}: {e}", tool.name()));
            let mut argv = vec![prepared.subcommand.to_string()];
            argv.extend(prepared.argv);
            parses(tool.name(), &argv);
        }
    }
}
