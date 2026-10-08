//! `pmetal decide` — answer one `/v1/systemone` request with a decision model.

use crate::{FieldError, JobFields};
use pmetal_core_derive::JobSpec;
use serde::{Deserialize, Serialize};

/// Spec for `pmetal decide`.
#[derive(Debug, Clone, Serialize, Deserialize, JobSpec)]
#[spec(kind = "Decide", subcommand = "decide")]
#[serde(rename_all = "snake_case")]
pub struct DecideSpec {
    #[job(
        label = "Model",
        group = "Model",
        argv = "--model",
        kind = "model_picker",
        required
    )]
    #[serde(default)]
    pub model: String,

    #[job(
        label = "Request JSON",
        group = "Input",
        argv = "--request",
        kind = "path",
        required
    )]
    #[serde(default)]
    pub request: String,

    #[job(
        label = "Max Prompt Tokens",
        group = "Input",
        argv = "--max-length",
        min = 1,
        max = 1_048_576,
        default_int = 16384
    )]
    #[serde(default = "default_max_length")]
    pub max_length: usize,

    #[job(
        label = "Compact Output",
        group = "Output",
        argv = "--compact",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub compact: bool,
}

impl Default for DecideSpec {
    fn default() -> Self {
        Self {
            model: String::new(),
            request: String::new(),
            max_length: default_max_length(),
            compact: false,
        }
    }
}

impl DecideSpec {
    pub fn normalize(&mut self) -> Result<(), Vec<FieldError>> {
        let errs = self.validate_descriptors();
        if errs.is_empty() { Ok(()) } else { Err(errs) }
    }
}

/// The release reference encoder's default prompt budget.
fn default_max_length() -> usize {
    16384
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argv_round_trip() {
        let spec = DecideSpec {
            model: "Cloudflare/clef-flash".into(),
            request: "request.json".into(),
            compact: true,
            ..Default::default()
        };
        let argv = spec.to_argv();
        assert!(argv.contains(&"--model".to_string()));
        assert!(argv.contains(&"--request".to_string()));
        assert!(argv.contains(&"--compact".to_string()));
    }
}
