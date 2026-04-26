//! `pmetal fuse` — fuse a LoRA adapter into base weights.

use crate::{FieldError, JobFields};
use pmetal_core_derive::JobSpec;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize, JobSpec)]
#[spec(kind = "Fuse", subcommand = "fuse")]
#[serde(rename_all = "snake_case")]
pub struct FuseSpec {
    #[job(label = "Base Model", group = "Source", argv = "--model", kind = "model_picker", required)]
    #[serde(default)]
    pub model: String,

    #[job(label = "LoRA Adapter", group = "Source", argv = "--lora", kind = "path", required)]
    #[serde(default)]
    pub lora: String,

    #[job(label = "Output Dir", group = "Output", argv = "--output", kind = "path", required)]
    #[serde(default)]
    pub output: String,

    #[job(label = "Override α", group = "LoRA", argv = "--alpha", min = 0.0, max = 1024.0)]
    #[serde(default)]
    pub alpha: Option<f32>,

    #[job(label = "Override Rank", group = "LoRA", argv = "--rank", min = 1, max = 4096)]
    #[serde(default)]
    pub rank: Option<usize>,

    #[job(label = "Accurate (f64)", group = "Method", argv = "--accurate", flag, default_bool = false)]
    #[serde(default)]
    pub accurate: bool,

    #[job(label = "Low-Memory", group = "Method", argv = "--low-memory", flag, default_bool = false)]
    #[serde(default)]
    pub low_memory: bool,
}

impl FuseSpec {
    pub fn normalize(&mut self) -> Result<(), Vec<FieldError>> {
        let errs = self.validate_descriptors();
        if errs.is_empty() {
            Ok(())
        } else {
            Err(errs)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argv_round_trip() {
        let mut spec = FuseSpec::default();
        spec.model = "m".into();
        spec.lora = "l".into();
        spec.output = "o".into();
        let argv = spec.to_argv();
        assert!(argv.contains(&"--model".to_string()));
        assert!(argv.contains(&"--lora".to_string()));
    }
}
