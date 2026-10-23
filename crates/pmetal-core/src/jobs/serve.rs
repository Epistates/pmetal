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
        default = "127.0.0.1",
        always
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
        label = "CORS Origins",
        group = "Server",
        argv = "--cors-origin",
        repeat,
        help = "Browser origins whose pages may call the server, or * for any; none sends no CORS headers"
    )]
    #[serde(default)]
    pub cors_origin: Vec<String>,

    #[job(
        label = "Request Timeout (s)",
        group = "Server",
        argv = "--request-timeout",
        min = 0,
        max = 604_800,
        default_int = 1800,
        help = "Seconds one request may run before the server ends it; 0 for no limit"
    )]
    #[serde(default = "default_request_timeout")]
    pub request_timeout: u64,

    #[job(
        label = "Max Seq Len",
        group = "Inference",
        argv = "--max-seq-len",
        min = 256,
        max = 4_194_304,
        help = "Default: the model's context window, capped by device memory"
    )]
    #[serde(default)]
    pub max_seq_len: Option<usize>,

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
        label = "ANE Max Context",
        group = "Compute",
        argv = "--ane-max-seq-len",
        default_int = 4096
    )]
    #[serde(default = "default_ane_max_seq_len")]
    pub ane_max_seq_len: usize,

    #[job(
        label = "Draft Model",
        group = "Speculative decoding",
        argv = "--draft-model",
        kind = "model_picker",
        help = "A draft model that speeds decoding up without changing the output: a Gemma 4 assistant, or a DFlash draft model (with --ane, DFlash only)"
    )]
    #[serde(default)]
    pub draft_model: Option<String>,

    #[job(
        label = "Qwen MTP",
        group = "Speculative decoding",
        argv = "--mtp",
        flag,
        default_bool = false,
        help = "Draft with a Qwen3.5/3.6/3.8 model's multi-token-prediction head (the mtp.* weights bundled in the model)"
    )]
    #[serde(default)]
    pub mtp: bool,

    #[job(
        label = "MTP Checkpoint",
        group = "Speculative decoding",
        argv = "--mtp-model",
        kind = "path",
        help = "Qwen MTP checkpoint directory; defaults to the mtp.* weights bundled in the model"
    )]
    #[serde(default)]
    pub mtp_model: Option<String>,

    #[job(
        label = "Draft Tokens",
        group = "Speculative decoding",
        argv = "--draft-tokens",
        min = 1,
        max = 32,
        help = "Tokens a draft round proposes (default: 3 for MTP, the assistant's own setting, the DFlash block size)"
    )]
    #[serde(default)]
    pub draft_tokens: Option<usize>,

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

    #[job(
        label = "CB Exact",
        group = "Server",
        argv = "--cb-exact",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub cb_exact: bool,
}

impl Default for ServeSpec {
    fn default() -> Self {
        Self {
            model: String::new(),
            host: default_host(),
            port: default_port(),
            cors_origin: Vec::new(),
            request_timeout: default_request_timeout(),
            max_seq_len: None,
            experts_dir: None,
            fp8: false,
            kv_quant: None,
            no_kv_quant: false,
            kv_group_size: default_kv_group_size(),
            kv_turboquant: false,
            kv_turboquant_preset: None,
            ane: false,
            ane_max_seq_len: default_ane_max_seq_len(),
            draft_model: None,
            mtp: false,
            mtp_model: None,
            draft_tokens: None,
            continuous_batch: false,
            cb_max_slots: default_cb_slots(),
            cb_max_queue_depth: default_cb_queue(),
            cb_block_size: default_cb_block_size(),
            cb_max_blocks: 0,
            cb_exact: false,
        }
    }
}

impl ServeSpec {
    /// Run descriptor + cross-field validation.
    pub fn normalize(&mut self) -> Result<(), Vec<FieldError>> {
        let mut errs = self.validate_descriptors();
        let drafting = self.mtp || self.draft_model.is_some();
        if self.mtp && self.draft_model.is_some() {
            errs.push(FieldError::new(
                "draft_model",
                "use either --mtp (a Qwen model's own head) or --draft-model, not both",
            ));
        }
        if self.mtp_model.is_some() && !self.mtp {
            errs.push(FieldError::new("mtp_model", "needs --mtp"));
        }
        if self.draft_tokens.is_some() && !drafting {
            errs.push(FieldError::new(
                "draft_tokens",
                "needs --mtp or --draft-model",
            ));
        }
        if self.mtp && self.ane {
            errs.push(FieldError::new(
                "mtp",
                "MTP drafts on the GPU; the ANE engine drafts with a DFlash --draft-model",
            ));
        }
        if drafting && self.continuous_batch {
            errs.push(FieldError::new(
                "continuous_batch",
                "speculative decoding runs one request at a time, and continuous batching serves many: use one of them",
            ));
        }
        // On the GPU the target runs on the float KV cache a rewind needs.
        if drafting && !self.ane {
            if self.kv_quant.is_some_and(|bits| bits != 0) || self.kv_turboquant {
                errs.push(FieldError::new(
                    "kv_quant",
                    "speculative decoding runs the model on a float KV cache; drop --kv-quant and --kv-turboquant, or drop the drafter",
                ));
            }
        }
        if errs.is_empty() { Ok(()) } else { Err(errs) }
    }
}

fn default_host() -> String {
    "127.0.0.1".to_string()
}
fn default_port() -> u16 {
    8080
}
fn default_request_timeout() -> u64 {
    1800
}
fn default_kv_group_size() -> usize {
    64
}
fn default_ane_max_seq_len() -> usize {
    4096
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
            port: 9090,
            ..Default::default()
        };
        let argv = spec.to_argv();
        assert!(argv.contains(&"--model".to_string()));
        assert!(argv.contains(&"--port".to_string()));
        assert!(argv.contains(&"9090".to_string()));
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

    fn invalid(mut spec: ServeSpec) -> Vec<String> {
        spec.model = "m".into();
        match spec.normalize() {
            Ok(()) => Vec::new(),
            Err(errs) => errs.into_iter().map(|e| e.field.to_string()).collect(),
        }
    }

    #[test]
    fn a_drafter_runs_on_the_gpu_or_the_ane() {
        // On the GPU a draft model or the model's own MTP head drafts.
        for spec in [
            ServeSpec {
                draft_model: Some("d".into()),
                ..Default::default()
            },
            ServeSpec {
                mtp: true,
                mtp_model: Some("head".into()),
                draft_tokens: Some(4),
                ..Default::default()
            },
            // On the ANE, a DFlash draft model.
            ServeSpec {
                ane: true,
                draft_model: Some("d".into()),
                ..Default::default()
            },
        ] {
            assert!(invalid(spec.clone()).is_empty(), "{spec:?}");
        }
        assert_eq!(
            invalid(ServeSpec {
                mtp: true,
                ane: true,
                ..Default::default()
            }),
            ["mtp"]
        );
    }

    #[test]
    fn drafting_options_need_a_drafter_and_exclude_each_other() {
        assert_eq!(
            invalid(ServeSpec {
                mtp: true,
                draft_model: Some("d".into()),
                ..Default::default()
            }),
            ["draft_model"]
        );
        assert_eq!(
            invalid(ServeSpec {
                mtp_model: Some("head".into()),
                ..Default::default()
            }),
            ["mtp_model"]
        );
        assert_eq!(
            invalid(ServeSpec {
                draft_tokens: Some(3),
                ..Default::default()
            }),
            ["draft_tokens"]
        );
    }

    #[test]
    fn a_drafter_and_continuous_batching_are_one_or_the_other() {
        for spec in [
            ServeSpec {
                mtp: true,
                continuous_batch: true,
                ..Default::default()
            },
            ServeSpec {
                draft_model: Some("d".into()),
                continuous_batch: true,
                ..Default::default()
            },
        ] {
            assert_eq!(invalid(spec), ["continuous_batch"]);
        }
    }

    #[test]
    fn a_drafter_needs_the_float_kv_cache() {
        for spec in [
            ServeSpec {
                mtp: true,
                kv_quant: Some(4),
                ..Default::default()
            },
            ServeSpec {
                draft_model: Some("d".into()),
                kv_turboquant: true,
                ..Default::default()
            },
        ] {
            assert_eq!(invalid(spec), ["kv_quant"]);
        }
        // Float is what it asks for anyway.
        assert!(
            invalid(ServeSpec {
                mtp: true,
                kv_quant: Some(0),
                no_kv_quant: true,
                ..Default::default()
            })
            .is_empty()
        );
    }

    #[test]
    fn subcommand_and_kind() {
        assert_eq!(ServeSpec::subcommand(), "serve");
        assert_eq!(ServeSpec::job_kind(), crate::JobKind::Serve);
    }
}
