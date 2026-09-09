//! `pmetal serve` — OpenAI-compatible inference server.

use crate::{FieldError, JobFields};
use pmetal_core_derive::JobSpec;
use serde::{Deserialize, Serialize};

/// Spec for `pmetal serve`.
#[derive(Debug, Clone, Serialize, Deserialize, JobSpec)]
#[spec(kind = "Serve", subcommand = "serve")]
#[serde(rename_all = "snake_case")]
pub struct ServeSpec {
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
        label = "Host",
        group = "Server",
        argv = "--host",
        default = "127.0.0.1"
    )]
    #[serde(default = "default_host")]
    pub host: String,

    #[job(
        label = "Port",
        group = "Server",
        argv = "--port",
        min = 1,
        max = 65535,
        default_int = 8080
    )]
    #[serde(default = "default_port")]
    pub port: u16,

    #[job(
        label = "Max Seq Len",
        group = "Inference",
        argv = "--max-seq-len",
        default_int = 4096
    )]
    #[serde(default = "default_max_seq_len")]
    pub max_seq_len: usize,

    #[job(
        label = "Experts Dir",
        group = "Model",
        argv = "--experts-dir",
        kind = "path"
    )]
    #[serde(default)]
    pub experts_dir: Option<String>,

    #[job(
        label = "FP8 Weights",
        group = "Quantization",
        argv = "--fp8",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub fp8: bool,

    #[job(
        label = "KV Cache Bits",
        group = "Quantization",
        argv = "--kv-quant",
        help = "8 / 4 / 0 (fp16); omit for auto"
    )]
    #[serde(default)]
    pub kv_quant: Option<u8>,

    #[job(
        label = "Disable KV Quant",
        group = "Quantization",
        argv = "--no-kv-quant",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub no_kv_quant: bool,

    #[job(
        label = "KV Group Size",
        group = "Quantization",
        argv = "--kv-group-size",
        default_int = 64
    )]
    #[serde(default = "default_kv_group_size")]
    pub kv_group_size: usize,

    #[job(
        label = "TurboQuant KV",
        group = "Quantization",
        argv = "--kv-turboquant",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub kv_turboquant: bool,

    #[job(
        label = "TurboQuant Preset",
        group = "Quantization",
        argv = "--kv-turboquant-preset",
        kind = "enum",
        enum_options = ["q2_5", "q3_5"]
    )]
    #[serde(default)]
    pub kv_turboquant_preset: Option<String>,

    #[job(
        label = "Use ANE",
        group = "Compute",
        argv = "--ane",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub ane: bool,

    #[job(
        label = "ANE Max Seq Len",
        group = "Compute",
        argv = "--ane-max-seq-len",
        default_int = 1024
    )]
    #[serde(default = "default_ane_max_seq_len")]
    pub ane_max_seq_len: usize,

    #[job(
        label = "ANE Real-Time",
        group = "Compute",
        argv = "--ane-real-time",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub ane_real_time: bool,

    #[job(
        label = "Continuous Batch",
        group = "Server",
        argv = "--continuous-batch",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub continuous_batch: bool,

    #[job(
        label = "CB Max Slots",
        group = "Server",
        argv = "--cb-max-slots",
        min = 1,
        max = 1024,
        default_int = 8
    )]
    #[serde(default = "default_cb_slots")]
    pub cb_max_slots: usize,

    #[job(
        label = "CB Queue Depth",
        group = "Server",
        argv = "--cb-max-queue-depth",
        min = 1,
        max = 65535,
        default_int = 256
    )]
    #[serde(default = "default_cb_queue")]
    pub cb_max_queue_depth: usize,

    #[job(
        label = "CB Block Size",
        group = "Server",
        argv = "--cb-block-size",
        min = 1,
        max = 1048576,
        default_int = 32
    )]
    #[serde(default = "default_cb_block_size")]
    pub cb_block_size: usize,

    #[job(
        label = "CB Max Blocks",
        group = "Server",
        argv = "--cb-max-blocks",
        min = 0,
        max = 1048576,
        default_int = 0
    )]
    #[serde(default)]
    pub cb_max_blocks: usize,
}

impl Default for ServeSpec {
    fn default() -> Self {
        Self {
            model: String::new(),
            host: default_host(),
            port: default_port(),
            max_seq_len: default_max_seq_len(),
            experts_dir: None,
            fp8: false,
            kv_quant: None,
            no_kv_quant: false,
            kv_group_size: default_kv_group_size(),
            kv_turboquant: false,
            kv_turboquant_preset: None,
            ane: false,
            ane_max_seq_len: default_ane_max_seq_len(),
            ane_real_time: false,
            continuous_batch: false,
            cb_max_slots: default_cb_slots(),
            cb_max_queue_depth: default_cb_queue(),
            cb_block_size: default_cb_block_size(),
            cb_max_blocks: 0,
        }
    }
}

impl ServeSpec {
    /// Run descriptor validation.
    pub fn normalize(&mut self) -> Result<(), Vec<FieldError>> {
        let errs = self.validate_descriptors();
        if errs.is_empty() { Ok(()) } else { Err(errs) }
    }
}

fn default_host() -> String {
    "127.0.0.1".to_string()
}
fn default_port() -> u16 {
    8080
}
fn default_max_seq_len() -> usize {
    4096
}
fn default_kv_group_size() -> usize {
    64
}
fn default_ane_max_seq_len() -> usize {
    1024
}
fn default_cb_slots() -> usize {
    8
}
fn default_cb_queue() -> usize {
    256
}
fn default_cb_block_size() -> usize {
    32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid() {
        let spec = ServeSpec {
            model: "model".into(),
            ..Default::default()
        };
        assert!(spec.validate_descriptors().is_empty());
    }

    #[test]
    fn argv_emits_required_fields() {
        let spec = ServeSpec {
            model: "model".into(),
            ..Default::default()
        };
        let argv = spec.to_argv();
        assert!(argv.contains(&"--model".to_string()));
        assert!(argv.contains(&"--port".to_string()));
        assert!(argv.contains(&"8080".to_string()));
    }

    #[test]
    fn flags_omitted_by_default() {
        let spec = ServeSpec {
            model: "m".into(),
            ..Default::default()
        };
        let argv = spec.to_argv();
        assert!(!argv.contains(&"--fp8".to_string()));
        assert!(!argv.contains(&"--ane".to_string()));
        assert!(!argv.contains(&"--continuous-batch".to_string()));
    }

    #[test]
    fn subcommand_and_kind() {
        assert_eq!(ServeSpec::subcommand(), "serve");
        assert_eq!(ServeSpec::job_kind(), crate::JobKind::Serve);
    }
}
