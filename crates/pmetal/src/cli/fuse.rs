//! Clap argument struct for `pmetal fuse`.

use clap::Args;

/// Thin clap argument struct for `pmetal fuse`.
#[derive(Args, Debug)]
pub struct FuseArgs {
    /// Base model ID or path
    #[arg(short, long = "model")]
    pub model: String,

    /// LoRA adapter path (directory containing lora_weights.safetensors, or the file itself)
    #[arg(short, long = "lora")]
    pub lora: String,

    /// Output directory for the fused model
    #[arg(short, long = "output")]
    pub output: String,

    /// LoRA scaling alpha (default: auto-detect from adapter)
    #[arg(long = "alpha")]
    pub alpha: Option<f32>,

    /// LoRA rank (default: auto-detect from adapter)
    #[arg(long = "rank")]
    pub rank: Option<usize>,

    /// Use f64-accurate LoRA merge.
    #[arg(long = "accurate", default_value_t = false)]
    pub accurate: bool,

    /// Use tiled low-memory mode with the --accurate path.
    #[arg(long = "low-memory", default_value_t = false)]
    pub low_memory: bool,
}
