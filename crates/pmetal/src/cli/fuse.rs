//! Clap argument struct for `pmetal fuse`.

use clap::Args;
use pmetal_core::jobs::FuseSpec;

/// Fuse LoRA adapter weights into a base model and save as a complete model.
///
/// The fold runs in f64 over memory-mapped tensors and writes each file
/// atomically; per-expert adapters of mixture-of-experts models are folded
/// into the experts' weights.
#[derive(Args, Debug)]
pub struct FuseArgs {
    /// Base model ID or path
    #[arg(short, long = "model")]
    pub model: String,

    /// LoRA adapter path (directory containing adapter_model.safetensors, or the file itself)
    #[arg(short, long = "lora")]
    pub lora: String,

    /// Output directory for the fused model
    #[arg(short, long = "output")]
    pub output: String,

    /// LoRA scaling alpha (default: read from the adapter)
    #[arg(long = "alpha")]
    pub alpha: Option<f32>,

    /// LoRA rank (default: read from the adapter)
    #[arg(long = "rank")]
    pub rank: Option<usize>,

    /// Tile the fold (512 rows at a time) to bound memory on very large tensors
    #[arg(long = "low-memory")]
    pub low_memory: bool,
}

impl From<FuseArgs> for FuseSpec {
    fn from(a: FuseArgs) -> Self {
        FuseSpec {
            model: a.model,
            lora: a.lora,
            output: a.output,
            alpha: a.alpha,
            rank: a.rank,
            low_memory: a.low_memory,
        }
    }
}
