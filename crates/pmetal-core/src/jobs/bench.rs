//! `pmetal bench` — training/inference benchmarking.

use crate::{FieldError, JobFields};
use pmetal_core_derive::JobSpec;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, JobSpec)]
#[spec(kind = "Bench", subcommand = "bench")]
#[serde(rename_all = "snake_case")]
pub struct BenchSpec {
    #[job(
        label = "Model",
        group = "Model",
        argv = "--model",
        kind = "model_picker",
        default = "Qwen/Qwen3-0.6B"
    )]
    #[serde(default = "default_model")]
    pub model: String,

    #[job(
        label = "Batch Size",
        group = "Bench",
        argv = "--batch-size",
        min = 1,
        max = 4096,
        default_int = 1
    )]
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,

    #[job(
        label = "Seq Len",
        group = "Bench",
        argv = "--seq-len",
        min = 1,
        max = 1_048_576,
        default_int = 512
    )]
    #[serde(default = "default_seq_len")]
    pub seq_len: usize,
}

impl Default for BenchSpec {
    fn default() -> Self {
        Self {
            model: default_model(),
            batch_size: default_batch_size(),
            seq_len: default_seq_len(),
        }
    }
}

impl BenchSpec {
    pub fn normalize(&mut self) -> Result<(), Vec<FieldError>> {
        let errs = self.validate_descriptors();
        if errs.is_empty() { Ok(()) } else { Err(errs) }
    }
}

fn default_model() -> String {
    "Qwen/Qwen3-0.6B".to_string()
}
fn default_batch_size() -> usize {
    1
}
fn default_seq_len() -> usize {
    512
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_emit_argv() {
        // The defaults are the command's own, so nothing is passed.
        assert!(BenchSpec::default().to_argv().is_empty());
        let spec = BenchSpec {
            model: "m".into(),
            batch_size: 3,
            ..Default::default()
        };
        let argv = spec.to_argv();
        assert!(argv.contains(&"--model".to_string()));
        assert!(argv.contains(&"--batch-size".to_string()));
    }
}
