//! Clap argument struct for `pmetal eval`.

use clap::Args;
use pmetal_core::jobs::EvalSpec;

/// Thin clap argument struct for `pmetal eval`.
#[derive(Args, Debug)]
pub struct EvalArgs {
    /// Model ID or path
    #[arg(short, long = "model")]
    pub model: String,

    /// Dataset path (JSONL file)
    #[arg(short, long = "dataset")]
    pub dataset: String,

    /// LoRA adapter path (optional)
    #[arg(long = "lora")]
    pub lora: Option<String>,

    /// Maximum sequence length
    #[arg(long = "max-seq-len", default_value = "1024")]
    pub max_seq_len: usize,

    /// Number of samples to evaluate (0 = all)
    #[arg(long = "num-samples", default_value = "0")]
    pub num_samples: usize,

    /// Output as JSON
    #[arg(long = "json")]
    pub json: bool,
}

impl From<EvalArgs> for EvalSpec {
    fn from(a: EvalArgs) -> Self {
        EvalSpec {
            model: a.model,
            dataset: a.dataset,
            lora: a.lora,
            max_seq_len: a.max_seq_len,
            num_samples: a.num_samples,
            json: a.json,
        }
    }
}
