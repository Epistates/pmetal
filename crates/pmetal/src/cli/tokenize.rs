//! Clap argument struct for `pmetal tokenize`.

use clap::Args;

/// Thin clap argument struct for `pmetal tokenize`.
#[derive(Args, Debug)]
pub struct TokenizeArgs {
    /// Input JSONL file
    #[arg(short, long = "input")]
    pub input: String,

    /// Output directory for shard files
    #[arg(short, long = "output")]
    pub output: String,

    /// Tokenizer model ID or path (HuggingFace format)
    #[arg(short, long = "tokenizer")]
    pub tokenizer: String,

    /// JSONL column containing text (default: "text")
    #[arg(long = "text-column", default_value = "text")]
    pub text_column: String,

    /// Maximum documents per shard (default: 10000)
    #[arg(long = "docs-per-shard", default_value = "10000")]
    pub docs_per_shard: usize,
}
