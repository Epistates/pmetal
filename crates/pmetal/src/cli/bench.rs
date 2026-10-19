//! Clap argument struct for `pmetal bench`.

use clap::{Args, ValueEnum};

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
