//! Clap argument struct for `pmetal bench`.

use clap::Args;

/// Thin clap argument struct for `pmetal bench`.
#[derive(Args, Debug)]
pub struct BenchArgs {
    /// Model to benchmark
    #[arg(short, long = "model", default_value = "meta-llama/Llama-3.2-1B")]
    pub model: String,

    /// Batch size
    #[arg(short, long = "batch-size", default_value = "1")]
    pub batch_size: usize,

    /// Sequence length
    #[arg(short, long = "seq-len", default_value = "512")]
    pub seq_len: usize,
}
