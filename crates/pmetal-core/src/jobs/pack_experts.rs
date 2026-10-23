//! `pmetal pack-experts` — pack routed MoE expert weights into shard files.

use crate::{FieldError, JobFields};
use pmetal_core_derive::JobSpec;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, JobSpec)]
#[spec(kind = "PackExperts", subcommand = "pack-experts")]
#[serde(rename_all = "snake_case")]
pub struct PackExpertsSpec {
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
        label = "Output Dir",
        group = "Output",
        argv = "--output",
        kind = "path",
        default = "./packed_experts"
    )]
    #[serde(default = "default_output")]
    pub output: String,

    #[job(
        label = "Bits",
        group = "Quantization",
        argv = "--bits",
        min = 2,
        max = 8
    )]
    #[serde(default)]
    pub bits: Option<u8>,
}

impl Default for PackExpertsSpec {
    fn default() -> Self {
        Self {
            model: String::new(),
            output: default_output(),
            bits: None,
        }
    }
}

impl PackExpertsSpec {
    pub fn normalize(&mut self) -> Result<(), Vec<FieldError>> {
        let errs = self.validate_descriptors();
        if errs.is_empty() { Ok(()) } else { Err(errs) }
    }
}

fn default_output() -> String {
    "./packed_experts".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argv_round_trip() {
        let spec = PackExpertsSpec {
            model: "m".into(),
            output: "packed".into(),
            ..Default::default()
        };
        let argv = spec.to_argv();
        assert!(argv.contains(&"--model".to_string()));
        assert!(argv.contains(&"--output".to_string()));
    }
}
