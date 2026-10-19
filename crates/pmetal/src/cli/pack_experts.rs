//! Clap argument struct for `pmetal pack-experts`.

use clap::Args;
use pmetal_core::jobs::PackExpertsSpec;

/// Thin clap argument struct for `pmetal pack-experts`.
#[derive(Args, Debug)]
pub struct PackExpertsArgs {
    /// Model directory (containing config.json and safetensors)
    #[arg(short, long = "model")]
    pub model: String,

    /// Output directory for packed expert files
    #[arg(short, long = "output", default_value = "./packed_experts")]
    pub output: String,

    /// Quantization bit width (4 or 2)
    #[arg(short, long = "bits")]
    pub bits: Option<u8>,
}

impl From<PackExpertsArgs> for PackExpertsSpec {
    fn from(a: PackExpertsArgs) -> Self {
        PackExpertsSpec {
            model: a.model,
            output: a.output,
            bits: a.bits,
        }
    }
}
