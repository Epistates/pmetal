//! `pmetal dflash` — block-diffusion speculative decoding.

use crate::{FieldError, JobFields};
use pmetal_core_derive::JobSpec;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, JobSpec)]
#[spec(kind = "Dflash", subcommand = "dflash")]
#[serde(rename_all = "snake_case")]
pub struct DflashSpec {
    #[job(
        label = "Target Model",
        group = "Models",
        argv = "--target",
        kind = "model_picker",
        required
    )]
    #[serde(default)]
    pub target: String,

    #[job(
        label = "Draft Model",
        group = "Models",
        argv = "--draft",
        kind = "model_picker",
        required
    )]
    #[serde(default)]
    pub draft: String,

    #[job(label = "Prompt", group = "Input", argv = "--prompt", required)]
    #[serde(default)]
    pub prompt: String,

    #[job(
        label = "Max New Tokens",
        group = "Sampling",
        argv = "--max-new-tokens",
        min = 1,
        max = 1_048_576,
        default_int = 128
    )]
    #[serde(default = "default_max_new_tokens")]
    pub max_new_tokens: usize,

    #[job(
        label = "Temperature",
        group = "Sampling",
        argv = "--temperature",
        min = 0.0,
        max = 5.0,
        default_float = 0.0
    )]
    #[serde(default)]
    pub temperature: f32,

    #[job(
        label = "Speculative Tokens",
        group = "Compute",
        argv = "--speculative-tokens",
        min = 1,
        max = 64
    )]
    #[serde(default)]
    pub speculative_tokens: Option<usize>,

    #[job(
        label = "Draft FP8",
        group = "Compute",
        argv = "--draft-fp8",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub draft_fp8: bool,

    #[job(
        label = "JSON Output",
        group = "Output",
        argv = "--json",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub json: bool,

    #[job(
        label = "No Chat Template",
        group = "Input",
        argv = "--no-chat",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub no_chat: bool,

    #[job(
        label = "Tree Budget",
        group = "Compute",
        argv = "--tree-budget",
        min = 0,
        max = 256,
        default_int = 0
    )]
    #[serde(default)]
    pub tree_budget: usize,

    #[job(
        label = "Compare With Greedy",
        group = "Output",
        argv = "--compare-greedy",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub compare_greedy: bool,
}

impl Default for DflashSpec {
    fn default() -> Self {
        Self {
            target: String::new(),
            draft: String::new(),
            prompt: String::new(),
            max_new_tokens: default_max_new_tokens(),
            temperature: 0.0,
            speculative_tokens: None,
            draft_fp8: false,
            json: false,
            no_chat: false,
            tree_budget: 0,
            compare_greedy: false,
        }
    }
}

impl DflashSpec {
    pub fn normalize(&mut self) -> Result<(), Vec<FieldError>> {
        let errs = self.validate_descriptors();
        if errs.is_empty() { Ok(()) } else { Err(errs) }
    }
}

fn default_max_new_tokens() -> usize {
    128
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argv_round_trip() {
        let spec = DflashSpec {
            target: "t".into(),
            draft: "d".into(),
            prompt: "hi".into(),
            ..Default::default()
        };
        let argv = spec.to_argv();
        assert!(argv.contains(&"--target".to_string()));
        assert!(argv.contains(&"--draft".to_string()));
    }
}
