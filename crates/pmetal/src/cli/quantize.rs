//! Clap argument struct for `pmetal quantize`.

use clap::Args;

/// Thin clap argument struct for `pmetal quantize`.
#[derive(Args, Debug)]
pub struct QuantizeArgs {
    /// Source model path (Safetensors/HF)
    #[arg(short, long = "model")]
    pub model: String,

    /// Output GGUF file path
    #[arg(short, long = "output")]
    pub output: String,

    /// Path to Importance Matrix (imatrix.dat) for dynamic quantization
    #[arg(long = "imatrix")]
    pub imatrix: Option<String>,

    /// Quantization method
    #[arg(long = "method", value_enum, default_value = "dynamic")]
    pub method: crate::QuantizeMethod,

    /// LoRA adapter to fuse before quantizing (optional)
    #[arg(long = "lora")]
    pub lora: Option<String>,

    /// Use KL-divergence calibration for per-tensor quantization type selection.
    #[arg(long = "kl-calibrate")]
    pub kl_calibrate: bool,

    /// Target average bits per weight for KL calibration (e.g. 4.5).
    #[arg(long = "target-bpw")]
    pub target_bpw: Option<f32>,

    /// Quality-loss threshold for KL calibration (default: 0.01).
    #[arg(long = "kl-threshold", default_value = "0.01")]
    pub kl_threshold: f64,

    /// Output format: "gguf" (default) or "mlx".
    #[arg(long = "format", default_value = "gguf")]
    pub format: String,

    /// Default bit-width for MLX-format quantization (3, 4, 5, 6, or 8).
    #[arg(long = "bits", default_value_t = 4)]
    pub bits: i32,

    /// Group size for MLX-format quantization.
    #[arg(long = "group-size", default_value_t = 64)]
    pub group_size: i32,
}
