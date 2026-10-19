//! Clap argument structs for the commands that talk to the Hugging Face Hub:
//! `download`, `upload` and `search`.

use clap::{Args, ValueEnum};
use pmetal_hub::WeightFormat;

/// The weight format to fetch.
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightsArg {
    /// Safetensors, else GGUF
    Auto,
    /// Safetensors only
    Safetensors,
    /// One GGUF quantization
    Gguf,
    /// PyTorch `pytorch_model*.bin` pickles
    Pytorch,
}

impl From<WeightsArg> for WeightFormat {
    fn from(arg: WeightsArg) -> Self {
        match arg {
            WeightsArg::Auto => WeightFormat::Auto,
            WeightsArg::Safetensors => WeightFormat::Safetensors,
            WeightsArg::Gguf => WeightFormat::Gguf,
            WeightsArg::Pytorch => WeightFormat::PyTorch,
        }
    }
}

/// How search results are ordered.
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum SortArg {
    /// Most downloaded in the last 30 days
    Downloads,
    /// Most liked
    Likes,
    /// Trending
    Trending,
    /// Most recently updated
    Updated,
}

impl From<SortArg> for pmetal_hub::search::SearchSort {
    fn from(arg: SortArg) -> Self {
        match arg {
            SortArg::Downloads => Self::Downloads,
            SortArg::Likes => Self::Likes,
            SortArg::Trending => Self::Trending,
            SortArg::Updated => Self::Updated,
        }
    }
}

/// Download a model from HuggingFace.
///
/// Fetches the configs, tokenizer and one set of weights from a single
/// commit, then checks them. Authenticates with `HF_TOKEN`, or the token
/// stored by `hf auth login`. `HF_ENDPOINT` selects a mirror and
/// `HF_HUB_OFFLINE=1` uses only the local cache.
#[derive(Args, Debug, Clone)]
pub struct DownloadArgs {
    /// Model ID, optionally with a GGUF quantization: `org/model-GGUF:Q4_K_M`
    pub model: String,

    /// Branch, tag or commit (default: the cached copy, else the default branch)
    #[arg(long)]
    pub revision: Option<String>,

    /// Weight format to fetch
    #[arg(long, value_enum, default_value_t = WeightsArg::Auto)]
    pub format: WeightsArg,

    /// GGUF quantization to fetch (e.g. Q4_K_M, Q8_0); implies --format gguf
    #[arg(long, value_name = "QUANT")]
    pub quant: Option<String>,

    /// Also hash large files against the Hub's SHA-256 digests (reads them back from disk)
    #[arg(long)]
    pub verify_sha256: bool,

    /// Show the commit, the files and their size, then stop
    #[arg(long)]
    pub dry_run: bool,

    /// Use only the local cache; never touch the network
    #[arg(long)]
    pub offline: bool,
}

/// Publish a model directory to HuggingFace (creates the repo if needed).
///
/// Authenticates with `HF_TOKEN`, or the token stored by `hf auth login`.
#[derive(Args, Debug, Clone)]
pub struct UploadArgs {
    /// Local directory to publish (a fused model, an adapter, a quantized export)
    pub path: String,

    /// Target repository, as owner/name
    pub repo: String,

    /// Create the repository as private (no effect if it already exists)
    #[arg(long)]
    pub private: bool,

    /// Branch to commit to (default: the repository's default branch)
    #[arg(long)]
    pub revision: Option<String>,

    /// Commit message
    #[arg(short, long)]
    pub message: Option<String>,

    /// Open a pull request instead of committing to the branch
    #[arg(long)]
    pub create_pr: bool,

    /// Glob of files to leave out, relative to PATH (repeatable), e.g. "checkpoints/**"
    #[arg(long = "exclude", value_name = "GLOB")]
    pub exclude: Vec<String>,
}

/// Search HuggingFace Hub for models and show device fit.
#[derive(Args, Debug, Clone)]
pub struct SearchArgs {
    /// Search query (e.g. "qwen3 0.6B", "llama 8b")
    pub query: String,

    /// Maximum number of results
    #[arg(short, long, default_value = "15")]
    pub limit: usize,

    /// Download the first result that fits on your device
    #[arg(long)]
    pub download: bool,

    /// Show detailed fit analysis (memory breakdown, training estimates)
    #[arg(long)]
    pub detailed: bool,

    /// Output results as JSON
    #[arg(long)]
    pub json: bool,

    /// Only models with this tag (repeatable): gguf, mlx, moe, license:apache-2.0, ...
    #[arg(long = "tag", value_name = "TAG")]
    pub tags: Vec<String>,

    /// Only models from this user or organization
    #[arg(long)]
    pub author: Option<String>,

    /// Pipeline tag to search; `any` searches every kind of model
    #[arg(long, default_value = "text-generation")]
    pub pipeline_tag: String,

    /// Result order
    #[arg(long, value_enum, default_value_t = SortArg::Downloads)]
    pub sort: SortArg,
}
