//! `pmetal quantize` — model quantization (GGUF / MLX).

use crate::{FieldError, JobFields};
use pmetal_core_derive::JobSpec;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, JobSpec)]
#[spec(kind = "Quantize", subcommand = "quantize")]
#[serde(rename_all = "snake_case")]
pub struct QuantizeSpec {
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
        label = "Output Path",
        group = "Output",
        argv = "--output",
        kind = "path",
        required
    )]
    #[serde(default)]
    pub output: String,

    #[job(label = "IMatrix", group = "Method", argv = "--imatrix", kind = "path")]
    #[serde(default)]
    pub imatrix: Option<String>,

    #[job(label = "Method", group = "Method", argv = "--method", kind = "enum",
          enum_options = [
              "dynamic",
              "q1_0",
              "q2_k",
              "q3_k_s",
              "q3_k_m",
              "q3_k_l",
              "q4_0",
              "q4_1",
              "q4_k_s",
              "q4_k_m",
              "q5_0",
              "q5_1",
              "q5_k_s",
              "q5_k_m",
              "q6_k",
              "q8_0",
              "q8_1",
              "tq1_0",
              "tq2_0",
              "mxfp4",
              "nvfp4",
              "bf16",
              "f16",
              "f32"
          ],
          default = "dynamic")]
    #[serde(default = "default_method")]
    pub method: String,

    #[job(
        label = "LoRA Adapter",
        group = "Source",
        argv = "--lora",
        kind = "path"
    )]
    #[serde(default)]
    pub lora: Option<String>,

    #[job(
        label = "KL Calibration",
        group = "Method",
        argv = "--kl-calibrate",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub kl_calibrate: bool,

    #[job(
        label = "Target BPW",
        group = "Method",
        argv = "--target-bpw",
        min = 1.0,
        max = 16.0
    )]
    #[serde(default)]
    pub target_bpw: Option<f32>,

    #[job(
        label = "KL Threshold",
        group = "Method",
        argv = "--kl-threshold",
        min = 0.0,
        max = 1.0,
        default_float = 0.01
    )]
    #[serde(default = "default_kl_threshold")]
    pub kl_threshold: f64,

    #[job(label = "Format", group = "Output", argv = "--format", kind = "enum",
          enum_options = ["gguf", "mlx"], default = "gguf")]
    #[serde(default = "default_format")]
    pub format: String,

    #[job(label = "MLX Bits", group = "Output", argv = "--bits", default_int = 4)]
    #[serde(default = "default_bits")]
    pub bits: i32,

    #[job(
        label = "MLX Group Size",
        group = "Output",
        argv = "--group-size",
        default_int = 64
    )]
    #[serde(default = "default_group_size")]
    pub group_size: i32,
}

impl Default for QuantizeSpec {
    fn default() -> Self {
        Self {
            model: String::new(),
            output: String::new(),
            imatrix: None,
            method: default_method(),
            lora: None,
            kl_calibrate: false,
            target_bpw: None,
            kl_threshold: default_kl_threshold(),
            format: default_format(),
            bits: default_bits(),
            group_size: default_group_size(),
        }
    }
}

impl QuantizeSpec {
    pub fn normalize(&mut self) -> Result<(), Vec<FieldError>> {
        let errs = self.validate_descriptors();
        if errs.is_empty() { Ok(()) } else { Err(errs) }
    }
}

fn default_method() -> String {
    "dynamic".to_string()
}
fn default_kl_threshold() -> f64 {
    0.01
}
fn default_format() -> String {
    "gguf".to_string()
}
fn default_bits() -> i32 {
    4
}
fn default_group_size() -> i32 {
    64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argv_round_trip() {
        let spec = QuantizeSpec {
            model: "m".into(),
            output: "out.gguf".into(),
            ..Default::default()
        };
        let argv = spec.to_argv();
        assert!(argv.contains(&"--model".to_string()));
        assert!(argv.contains(&"--output".to_string()));
    }
}
