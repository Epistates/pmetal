//! `pmetal imatrix` — measure an importance matrix on calibration data.

use crate::{FieldError, JobFields};
use pmetal_core_derive::JobSpec;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, JobSpec)]
#[spec(kind = "Imatrix", subcommand = "imatrix")]
#[serde(rename_all = "snake_case")]
pub struct ImatrixSpec {
    #[job(
        label = "Model",
        group = "Source",
        argv = "--model",
        kind = "model_picker",
        required
    )]
    #[serde(default)]
    pub model: String,

    #[job(
        label = "Calibration Data",
        group = "Data",
        argv = "--dataset",
        kind = "dataset_picker",
        help = "A text file (one document), or a JSONL / Parquet dataset: a local path or a Hub dataset id",
        required
    )]
    #[serde(default)]
    pub dataset: String,

    #[job(
        label = "Output Path",
        group = "Output",
        argv = "--output",
        kind = "path",
        help = "A .gguf file (the imatrix format of current tools) or a .dat file (the legacy format)",
        default = "./imatrix.gguf"
    )]
    #[serde(default = "default_output")]
    pub output: String,

    #[job(
        label = "Context Size",
        group = "Calibration",
        argv = "--ctx-size",
        help = "Tokens per chunk: the context each forward runs at",
        min = 8,
        max = 131072,
        default_int = 512
    )]
    #[serde(default = "default_ctx_size")]
    pub ctx_size: usize,

    #[job(
        label = "Chunks",
        group = "Calibration",
        argv = "--chunks",
        help = "Chunks to run (0 = all the data makes)",
        default_int = 0
    )]
    #[serde(default)]
    pub chunks: usize,

    #[job(
        label = "Batch Size",
        group = "Calibration",
        argv = "--batch-size",
        help = "Chunks per forward pass",
        min = 1,
        max = 256,
        default_int = 4
    )]
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
}

impl Default for ImatrixSpec {
    fn default() -> Self {
        Self {
            model: String::new(),
            dataset: String::new(),
            output: default_output(),
            ctx_size: default_ctx_size(),
            chunks: 0,
            batch_size: default_batch_size(),
        }
    }
}

impl ImatrixSpec {
    pub fn normalize(&mut self) -> Result<(), Vec<FieldError>> {
        let errs = self.validate_descriptors();
        if errs.is_empty() { Ok(()) } else { Err(errs) }
    }
}

fn default_output() -> String {
    "./imatrix.gguf".to_string()
}
fn default_ctx_size() -> usize {
    512
}
fn default_batch_size() -> usize {
    4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argv_round_trip() {
        let spec = ImatrixSpec {
            model: "m".into(),
            dataset: "d.jsonl".into(),
            ..Default::default()
        };
        let argv = spec.to_argv();
        for flag in [
            "--model",
            "--dataset",
            "--output",
            "--ctx-size",
            "--batch-size",
        ] {
            assert!(argv.contains(&flag.to_string()), "{flag} in {argv:?}");
        }
    }

    #[test]
    fn the_model_and_the_data_are_required() {
        let mut spec = ImatrixSpec::default();
        let errors = spec.normalize().unwrap_err();
        assert_eq!(errors.len(), 2, "{errors:?}");
    }
}
