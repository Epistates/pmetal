//! Clap argument struct for `pmetal imatrix`.

use clap::Args;
use pmetal_core::jobs::ImatrixSpec;

/// Thin clap argument struct for `pmetal imatrix`.
#[derive(Args, Debug)]
pub struct ImatrixArgs {
    /// Model to measure (Hub id or local Safetensors directory)
    #[arg(short, long = "model")]
    pub model: String,

    /// Calibration data: a text file (one document), or a JSONL / Parquet
    /// dataset, as a local path or a Hub dataset id
    #[arg(short, long = "dataset")]
    pub dataset: String,

    /// Where to write the importance matrix: a .gguf file (the format of
    /// current tools) or a .dat file (the legacy format)
    #[arg(short, long = "output", default_value = "./imatrix.gguf")]
    pub output: String,

    /// Tokens per chunk: the context each forward runs at
    #[arg(long = "ctx-size", default_value_t = 512)]
    pub ctx_size: usize,

    /// Chunks to run (0 = all the data makes)
    #[arg(long = "chunks", default_value_t = 0)]
    pub chunks: usize,

    /// Chunks per forward pass
    #[arg(long = "batch-size", default_value_t = 4)]
    pub batch_size: usize,
}

impl From<ImatrixArgs> for ImatrixSpec {
    fn from(a: ImatrixArgs) -> Self {
        ImatrixSpec {
            model: a.model,
            dataset: a.dataset,
            output: a.output,
            ctx_size: a.ctx_size,
            chunks: a.chunks,
            batch_size: a.batch_size,
        }
    }
}
