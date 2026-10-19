//! Clap argument struct for `pmetal quantize`.

use clap::{Args, ValueEnum};

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
    pub method: crate::cli::quantize::QuantizeMethod,

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

/// GGUF quantization method for the `quantize` subcommand.
///
/// Supports all K-quant types available in the pmetal-gguf crate, plus
/// `dynamic` for importance-matrix-guided mixed-precision quantization.
/// K-quant size suffixes follow the GGUF K-quant naming convention:
///   - `s` = small (lower quality, smaller size)
///   - `m` = medium (balanced, recommended)
///   - `l` = large (higher quality, larger size)
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
pub enum QuantizeMethod {
    /// Importance-matrix-guided mixed precision (recommended with --imatrix)
    #[default]
    #[value(name = "dynamic")]
    Dynamic,
    /// 8-bit integer (near-lossless, ~1.06 bpw)
    #[value(name = "q8_0")]
    Q8_0,
    /// 8-bit integer with dot-product sum helper
    #[value(name = "q8_1")]
    Q8_1,
    /// 6-bit K-quant (high quality, ~0.80 bpw)
    #[value(name = "q6_k")]
    Q6K,
    /// 5-bit K-quant medium (good quality/size balance)
    #[value(name = "q5_k_m")]
    Q5KM,
    /// 5-bit K-quant small (slightly smaller than q5-k-m)
    #[value(name = "q5_k_s")]
    Q5KS,
    /// Legacy 5-bit symmetric GGML quantization
    #[value(name = "q5_0")]
    Q5_0,
    /// Legacy 5-bit affine GGML quantization
    #[value(name = "q5_1")]
    Q5_1,
    /// 4-bit K-quant medium (recommended 4-bit, ~0.58 bpw)
    #[value(name = "q4_k_m")]
    Q4KM,
    /// 4-bit K-quant small (smallest 4-bit K-quant)
    #[value(name = "q4_k_s")]
    Q4KS,
    /// Legacy 4-bit symmetric GGML quantization
    #[value(name = "q4_0")]
    Q4_0,
    /// Legacy 4-bit affine GGML quantization
    #[value(name = "q4_1")]
    Q4_1,
    /// 3-bit K-quant medium
    #[value(name = "q3_k_m")]
    Q3KM,
    /// 3-bit K-quant small
    #[value(name = "q3_k_s")]
    Q3KS,
    /// 3-bit K-quant large
    #[value(name = "q3_k_l")]
    Q3KL,
    /// 2-bit K-quant (lowest quality, ~0.37 bpw)
    #[value(name = "q2_k")]
    Q2K,
    /// 1-bit sign GGML quantization
    #[value(name = "q1_0")]
    Q1_0,
    /// Ternary 1.69-bit GGML quantization
    #[value(name = "tq1_0")]
    Tq1_0,
    /// Ternary 2.06-bit GGML quantization
    #[value(name = "tq2_0")]
    Tq2_0,
    /// MXFP4 block-floating GGML quantization
    #[value(name = "mxfp4")]
    Mxfp4,
    /// NVFP4 block-floating GGML quantization
    #[value(name = "nvfp4")]
    Nvfp4,
    /// BFloat16 (lossless-ish dense export, 2.0 bpw)
    #[value(name = "bf16")]
    Bf16,
    /// 16-bit float (lossless, 2.0 bpw)
    #[value(name = "f16")]
    F16,
    /// 32-bit float (lossless reference precision, 4.0 bpw)
    #[value(name = "f32")]
    F32,
}

impl QuantizeMethod {
    /// Canonical display name matching the GGUF K-quant naming convention.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Dynamic => "dynamic",
            Self::Q8_0 => "q8_0",
            Self::Q8_1 => "q8_1",
            Self::Q6K => "q6_k",
            Self::Q5KM => "q5_k_m",
            Self::Q5KS => "q5_k_s",
            Self::Q5_0 => "q5_0",
            Self::Q5_1 => "q5_1",
            Self::Q4KM => "q4_k_m",
            Self::Q4KS => "q4_k_s",
            Self::Q4_0 => "q4_0",
            Self::Q4_1 => "q4_1",
            Self::Q3KM => "q3_k_m",
            Self::Q3KS => "q3_k_s",
            Self::Q3KL => "q3_k_l",
            Self::Q2K => "q2_k",
            Self::Q1_0 => "q1_0",
            Self::Tq1_0 => "tq1_0",
            Self::Tq2_0 => "tq2_0",
            Self::Mxfp4 => "mxfp4",
            Self::Nvfp4 => "nvfp4",
            Self::Bf16 => "bf16",
            Self::F16 => "f16",
            Self::F32 => "f32",
        }
    }

    /// Map to the underlying `GgmlType`. Returns `None` for `Dynamic` (uses
    /// `DynamicQuantizer` instead of a fixed type).
    pub fn to_ggml_type(self) -> Option<pmetal_gguf::GgmlType> {
        use pmetal_gguf::GgmlType;
        match self {
            Self::Dynamic => None,
            Self::Q8_0 => Some(GgmlType::Q8_0),
            Self::Q8_1 => Some(GgmlType::Q8_1),
            Self::Q6K => Some(GgmlType::Q6K),
            // All Q5K size variants share the same K-quant block structure
            Self::Q5KM | Self::Q5KS => Some(GgmlType::Q5K),
            Self::Q5_0 => Some(GgmlType::Q5_0),
            Self::Q5_1 => Some(GgmlType::Q5_1),
            // All Q4K size variants share the same K-quant block structure
            Self::Q4KM | Self::Q4KS => Some(GgmlType::Q4K),
            Self::Q4_0 => Some(GgmlType::Q4_0),
            Self::Q4_1 => Some(GgmlType::Q4_1),
            // All Q3K size variants share the same K-quant block structure
            Self::Q3KM | Self::Q3KS | Self::Q3KL => Some(GgmlType::Q3K),
            Self::Q2K => Some(GgmlType::Q2K),
            Self::Q1_0 => Some(GgmlType::Q1_0),
            Self::Tq1_0 => Some(GgmlType::Tq1_0),
            Self::Tq2_0 => Some(GgmlType::Tq2_0),
            Self::Mxfp4 => Some(GgmlType::Mxfp4),
            Self::Nvfp4 => Some(GgmlType::Nvfp4),
            Self::Bf16 => Some(GgmlType::Bf16),
            Self::F16 => Some(GgmlType::F16),
            Self::F32 => Some(GgmlType::F32),
        }
    }
}
