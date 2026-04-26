//! Clap argument struct for `pmetal serve`.

use clap::Args;

/// Thin clap argument struct for `pmetal serve`.
#[derive(Args, Debug)]
pub struct ServeArgs {
    /// Model ID or path
    #[arg(short, long = "model")]
    pub model: String,

    /// Port to listen on
    #[arg(short, long = "port", default_value = "8080")]
    pub port: u16,

    /// Host to bind to
    #[arg(long = "host", default_value = "127.0.0.1")]
    pub host: String,

    /// Maximum sequence length for KV cache
    #[arg(long = "max-seq-len", default_value = "4096")]
    pub max_seq_len: usize,

    /// Path to packed expert weights directory for SSD-offloaded MoE inference.
    #[arg(long = "experts-dir")]
    pub experts_dir: Option<String>,

    /// Quantize weights to FP8 E4M3 at load time (~2x memory savings).
    #[arg(long = "fp8")]
    pub fp8: bool,

    /// KV cache quantization bits: 8 = q8_0, 4 = q4_0, 0 = fp16.
    #[arg(long = "kv-quant")]
    pub kv_quant: Option<u8>,

    /// Disable KV cache quantization (use fp16 KV cache).
    #[arg(long = "no-kv-quant")]
    pub no_kv_quant: bool,

    /// Quantization group size for KV cache.
    #[arg(long = "kv-group-size", default_value = "64")]
    pub kv_group_size: usize,

    /// Enable TurboQuant KV cache compression.
    #[arg(long = "kv-turboquant")]
    pub kv_turboquant: bool,

    /// TurboQuant quality preset: q2_5 or q3_5.
    #[arg(long = "kv-turboquant-preset", value_parser = ["q2_5", "q3_5"])]
    pub kv_turboquant_preset: Option<String>,

    /// Enable ANE (Apple Neural Engine) for serving (experimental).
    #[cfg(feature = "ane")]
    #[arg(long = "ane")]
    pub ane: bool,

    /// Maximum ANE kernel sequence length (power-of-2 bucket cap).
    #[cfg(feature = "ane")]
    #[arg(long = "ane-max-seq-len", default_value = "1024")]
    pub ane_max_seq_len: usize,

    /// Use the experimental ANE real-time evaluation path when ANE serving is selected.
    #[cfg(feature = "ane")]
    #[arg(long = "ane-real-time")]
    pub ane_real_time: bool,

    /// Enable continuous batching.
    #[arg(long = "continuous-batch")]
    pub continuous_batch: bool,

    /// Maximum concurrent slots when --continuous-batch is set.
    #[arg(long = "cb-max-slots", default_value = "8")]
    pub cb_max_slots: usize,

    /// Maximum pending-queue depth when --continuous-batch is set.
    #[arg(long = "cb-max-queue-depth", default_value = "256")]
    pub cb_max_queue_depth: usize,
}
