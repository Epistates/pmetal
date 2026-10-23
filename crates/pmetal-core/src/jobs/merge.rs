//! `pmetal merge`: merge two models (SLERP, TIES, DARE, task arithmetic, ...).
//!
//! The `method` field's options are the one list of merge methods every
//! surface offers (the command line's `--method`, the forms, the MCP tool, the
//! Python class). Fisher-weighted and RegMean merging are in the library but
//! need a statistics file per model that nothing in PMetal produces, so they
//! are not offered here.

use crate::{FieldError, JobFields};
use pmetal_core_derive::JobSpec;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, JobSpec)]
#[spec(kind = "Merge", subcommand = "merge")]
#[serde(rename_all = "snake_case")]
pub struct MergeSpec {
    #[job(
        label = "Model A",
        group = "Models",
        argv = "--model-a",
        kind = "model_picker",
        required
    )]
    #[serde(default)]
    pub model_a: String,

    #[job(
        label = "Model B",
        group = "Models",
        argv = "--model-b",
        kind = "model_picker",
        required
    )]
    #[serde(default)]
    pub model_b: String,

    #[job(
        label = "Output Dir",
        group = "Output",
        argv = "--output",
        kind = "path",
        required
    )]
    #[serde(default)]
    pub output: String,

    #[job(label = "Method", group = "Method", argv = "--method", kind = "enum",
          help = "Every method but linear, slerp, multislerp and passthrough merges what each model changed from a base model, which it then needs",
          enum_options = ["linear", "slerp", "ties", "dare_ties", "dare_linear",
                          "task_arithmetic", "della", "della_linear", "breadcrumbs",
                          "model_stock", "consensus", "multislerp", "nearswap", "ram",
                          "ram_plus", "passthrough"],
          default = "slerp")]
    #[serde(default = "default_method")]
    pub method: String,

    #[job(
        label = "Base Model",
        group = "Models",
        argv = "--base",
        kind = "model_picker"
    )]
    #[serde(default)]
    pub base: Option<String>,

    #[job(
        label = "SLERP t",
        group = "Method",
        argv = "--t",
        min = 0.0,
        max = 1.0,
        default_float = 0.5
    )]
    #[serde(default = "default_t")]
    pub t: f32,

    #[job(
        label = "Weight A",
        group = "Method",
        argv = "--weight-a",
        min = 0.0,
        max = 10.0,
        default_float = 0.5
    )]
    #[serde(default = "default_weight")]
    pub weight_a: f32,

    #[job(
        label = "Weight B",
        group = "Method",
        argv = "--weight-b",
        min = 0.0,
        max = 10.0,
        default_float = 0.5
    )]
    #[serde(default = "default_weight")]
    pub weight_b: f32,

    #[job(
        label = "Density",
        group = "Method",
        argv = "--density",
        min = 0.0,
        max = 1.0,
        default_float = 0.5
    )]
    #[serde(default = "default_density")]
    pub density: f32,

    #[job(label = "Output Dtype", group = "Output", argv = "--dtype", kind = "enum",
          enum_options = ["float32", "float16", "bfloat16"], default = "bfloat16")]
    #[serde(default = "default_dtype")]
    pub dtype: String,
}

impl Default for MergeSpec {
    fn default() -> Self {
        Self {
            model_a: String::new(),
            model_b: String::new(),
            output: String::new(),
            method: default_method(),
            base: None,
            t: default_t(),
            weight_a: default_weight(),
            weight_b: default_weight(),
            density: default_density(),
            dtype: default_dtype(),
        }
    }
}

/// The methods that merge the two models as they are; every other method
/// merges what each changed from a common base model.
pub const METHODS_WITHOUT_BASE: &[&str] = &["linear", "slerp", "multislerp", "passthrough"];

impl MergeSpec {
    pub fn normalize(&mut self) -> Result<(), Vec<FieldError>> {
        let mut errs = self.validate_descriptors();
        let has_base = self.base.as_deref().is_some_and(|b| !b.trim().is_empty());
        if errs.is_empty() && !has_base && !METHODS_WITHOUT_BASE.contains(&self.method.as_str()) {
            errs.push(FieldError::new(
                "base",
                format!(
                    "{} merges what each model changed from a base model: give the model both were trained from",
                    self.method
                ),
            ));
        }
        if errs.is_empty() { Ok(()) } else { Err(errs) }
    }
}

fn default_method() -> String {
    "slerp".to_string()
}
fn default_t() -> f32 {
    0.5
}
fn default_weight() -> f32 {
    0.5
}
fn default_density() -> f32 {
    0.5
}
fn default_dtype() -> String {
    "bfloat16".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_method_is_offered_and_the_base_rule_covers_them_all() {
        let options = match MergeSpec::field_descriptors()
            .iter()
            .find(|d| d.name == "method")
            .map(|d| d.kind)
        {
            Some(crate::FieldKind::Enum { options }) => options,
            other => panic!("method is not an enum: {other:?}"),
        };
        assert_eq!(options.len(), 16);
        for method in METHODS_WITHOUT_BASE {
            assert!(options.contains(method), "{method}");
        }
        for method in options {
            let mut spec = MergeSpec {
                model_a: "a".into(),
                model_b: "b".into(),
                output: "o".into(),
                method: (*method).into(),
                ..Default::default()
            };
            let needs_base = !METHODS_WITHOUT_BASE.contains(method);
            let errors = spec.normalize().err().unwrap_or_default();
            assert_eq!(
                errors.iter().any(|e| e.field == "base"),
                needs_base,
                "{method}: {errors:?}"
            );
            spec.base = Some("ancestor".into());
            assert!(spec.normalize().is_ok(), "{method}");
        }
    }

    #[test]
    fn argv_round_trip() {
        let spec = MergeSpec {
            model_a: "a".into(),
            model_b: "b".into(),
            output: "o".into(),
            ..Default::default()
        };
        let argv = spec.to_argv();
        assert!(argv.contains(&"--model-a".to_string()));
        assert!(argv.contains(&"--model-b".to_string()));
        assert!(argv.contains(&"--output".to_string()));
    }
}
