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

    #[job(
        label = "IMatrix",
        group = "Method",
        argv = "--imatrix",
        kind = "path",
        help = "An importance matrix file (GGUF or legacy .dat), from `pmetal imatrix`"
    )]
    #[serde(default)]
    pub imatrix: Option<String>,

    #[job(
        label = "Calibration Data",
        group = "Method",
        argv = "--calibration-dataset",
        kind = "dataset_picker",
        help = "Measure the importance matrix on this data first (a text file, or a JSONL / Parquet dataset), instead of reading one with --imatrix"
    )]
    #[serde(default)]
    pub calibration_dataset: Option<String>,

    #[job(
        label = "Calibration Context",
        group = "Method",
        argv = "--calibration-ctx-size",
        help = "Tokens per calibration chunk",
        min = 8,
        max = 131072,
        default_int = 512
    )]
    #[serde(default = "default_calibration_ctx_size")]
    pub calibration_ctx_size: usize,

    #[job(
        label = "Calibration Chunks",
        group = "Method",
        argv = "--calibration-chunks",
        help = "Calibration chunks to run (0 = all the data makes)",
        default_int = 0
    )]
    #[serde(default)]
    pub calibration_chunks: usize,

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
              "iq4_nl",
              "iq4_xs",
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
            calibration_dataset: None,
            calibration_ctx_size: default_calibration_ctx_size(),
            calibration_chunks: 0,
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
        let mut errs = self.validate_descriptors();
        if self.imatrix.is_some() && self.calibration_dataset.is_some() {
            errs.push(FieldError::new(
                "calibration_dataset",
                "give either an importance matrix file (--imatrix) or calibration data to \
                 measure one from (--calibration-dataset), not both",
            ));
        }
        if errs.is_empty() { Ok(()) } else { Err(errs) }
    }
}

fn default_calibration_ctx_size() -> usize {
    512
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

    /// A matrix is read or measured, not both.
    #[test]
    fn an_imatrix_file_and_calibration_data_are_exclusive() {
        let mut spec = QuantizeSpec {
            model: "m".into(),
            output: "o.gguf".into(),
            imatrix: Some("im.gguf".into()),
            ..Default::default()
        };
        assert!(spec.normalize().is_ok());
        spec.imatrix = None;
        spec.calibration_dataset = Some("calib.txt".into());
        assert!(spec.normalize().is_ok());
        spec.imatrix = Some("im.gguf".into());
        let errors = spec.normalize().unwrap_err();
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert_eq!(errors[0].field, "calibration_dataset");
    }

    /// The two 4-bit non-linear formats are methods.
    #[test]
    fn the_iq4_formats_are_quantize_methods() {
        for method in ["iq4_nl", "iq4_xs"] {
            let spec = QuantizeSpec {
                model: "m".into(),
                output: "o.gguf".into(),
                method: method.into(),
                ..Default::default()
            };
            let mut spec = spec;
            assert!(spec.normalize().is_ok(), "{method}");
        }
    }
}
