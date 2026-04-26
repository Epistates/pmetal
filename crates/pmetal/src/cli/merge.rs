//! Clap argument struct for `pmetal merge`.

use clap::Args;

/// Thin clap argument struct for `pmetal merge`.
#[derive(Args, Debug)]
pub struct MergeArgs {
    /// First model path or HuggingFace ID
    #[arg(short = 'a', long = "model-a")]
    pub model_a: String,

    /// Second model path or HuggingFace ID
    #[arg(short = 'b', long = "model-b")]
    pub model_b: String,

    /// Output directory for merged model
    #[arg(short, long = "output")]
    pub output: String,

    /// Merge method (linear, slerp, ties, dare_ties, dare_linear, task_arithmetic, della, breadcrumbs, model_stock, nearswap, passthrough)
    #[arg(long = "method", default_value = "slerp")]
    pub method: String,

    /// Base model for task-vector methods (TIES, DARE, task_arithmetic)
    #[arg(long = "base")]
    pub base: Option<String>,

    /// Interpolation parameter t for SLERP (0.0=model_a, 1.0=model_b)
    #[arg(long = "t", default_value = "0.5")]
    pub t: f32,

    /// Weight for model_a in linear/ties methods
    #[arg(long = "weight-a", default_value = "0.5")]
    pub weight_a: f32,

    /// Weight for model_b in linear/ties methods
    #[arg(long = "weight-b", default_value = "0.5")]
    pub weight_b: f32,

    /// Density for sparsification (TIES/DARE) — fraction of params to keep
    #[arg(long = "density", default_value = "0.5")]
    pub density: f32,

    /// Output dtype (float32, float16, bfloat16)
    #[arg(long = "dtype", default_value = "bfloat16")]
    pub dtype: String,
}
