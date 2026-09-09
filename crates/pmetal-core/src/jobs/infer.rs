//! `pmetal infer` — one-shot or chat inference.

use crate::{FieldError, JobFields};
use pmetal_core_derive::JobSpec;
use serde::{Deserialize, Serialize};

/// Spec for `pmetal infer`.
#[derive(Debug, Clone, Serialize, Deserialize, JobSpec)]
#[spec(kind = "Infer", subcommand = "infer")]
#[serde(rename_all = "snake_case")]
pub struct InferSpec {
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
        label = "LoRA Adapter",
        group = "Model",
        argv = "--lora",
        kind = "path"
    )]
    #[serde(default)]
    pub lora: Option<String>,

    #[job(label = "Prompt", group = "Input", argv = "--prompt", required)]
    #[serde(default)]
    pub prompt: String,

    #[job(
        label = "Max Tokens",
        group = "Sampling",
        argv = "--max-tokens",
        min = 1,
        max = 1_048_576,
        default_int = 256
    )]
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,

    #[job(
        label = "Temperature",
        group = "Sampling",
        argv = "--temperature",
        min = 0.0,
        max = 5.0
    )]
    #[serde(default)]
    pub temperature: Option<f32>,

    #[job(
        label = "Top-k",
        group = "Sampling",
        argv = "--top-k",
        min = 0,
        max = 1_000_000
    )]
    #[serde(default)]
    pub top_k: Option<usize>,

    #[job(
        label = "Top-p",
        group = "Sampling",
        argv = "--top-p",
        min = 0.0,
        max = 1.0
    )]
    #[serde(default)]
    pub top_p: Option<f32>,

    #[job(
        label = "Min-p",
        group = "Sampling",
        argv = "--min-p",
        min = 0.0,
        max = 1.0
    )]
    #[serde(default)]
    pub min_p: Option<f32>,

    #[job(
        label = "Repetition Penalty",
        group = "Sampling",
        argv = "--repetition-penalty",
        min = 0.0,
        max = 10.0
    )]
    #[serde(default)]
    pub repetition_penalty: Option<f32>,

    #[job(label = "Frequency Penalty", group = "Sampling", argv = "--frequency-penalty", min = -10.0, max = 10.0)]
    #[serde(default)]
    pub frequency_penalty: Option<f32>,

    #[job(label = "Presence Penalty", group = "Sampling", argv = "--presence-penalty", min = -10.0, max = 10.0)]
    #[serde(default)]
    pub presence_penalty: Option<f32>,

    #[job(label = "Seed", group = "Sampling", argv = "--seed")]
    #[serde(default)]
    pub seed: Option<u64>,

    #[job(
        label = "Chat Mode",
        group = "Input",
        argv = "--chat",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub chat: bool,

    #[job(label = "System Message", group = "Input", argv = "--system")]
    #[serde(default)]
    pub system: Option<String>,

    #[job(
        label = "No Thinking",
        group = "Input",
        argv = "--no-thinking",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub no_thinking: bool,

    #[job(
        label = "Hide Thinking",
        group = "Output",
        argv = "--hide-thinking",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub hide_thinking: bool,

    #[job(
        label = "Sampling Mode",
        group = "Sampling",
        argv = "--mode",
        kind = "enum",
        enum_options = ["auto", "thinking-general", "thinking-coding", "instruct-general", "instruct-reasoning"],
        default = "auto"
    )]
    #[serde(default = "default_mode")]
    pub mode: String,

    #[job(
        label = "Backend",
        group = "Compute",
        argv = "--backend",
        kind = "enum",
        enum_options = ["auto", "standard", "compiled", "metal-sampler", "ane", "minimal", "dflash"],
        default = "auto"
    )]
    #[serde(default = "default_backend")]
    pub backend: String,

    #[job(
        label = "Draft Model",
        group = "Compute",
        argv = "--draft-model",
        kind = "model_picker"
    )]
    #[serde(default)]
    pub draft_model: Option<String>,

    #[job(
        label = "Compiled Sampling",
        group = "Compute",
        argv = "--compiled",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub compiled: bool,

    #[job(
        label = "Metal Sampler",
        group = "Compute",
        argv = "--metal-sampler",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub metal_sampler: bool,

    #[job(
        label = "Minimal Path",
        group = "Compute",
        argv = "--minimal",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub minimal: bool,

    #[job(label = "Tools JSON", group = "Input", argv = "--tools", kind = "path")]
    #[serde(default)]
    pub tools: Option<String>,

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
        label = "Experts Dir",
        group = "Model",
        argv = "--experts-dir",
        kind = "path"
    )]
    #[serde(default)]
    pub experts_dir: Option<String>,

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
        label = "Benchmark",
        group = "Benchmark",
        argv = "--benchmark",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub benchmark: bool,

    #[job(
        label = "Benchmark Iters",
        group = "Benchmark",
        argv = "--benchmark-iters",
        min = 1,
        max = 1_000_000,
        default_int = 5
    )]
    #[serde(default = "default_benchmark_iters")]
    pub benchmark_iters: usize,

    #[job(
        label = "Benchmark Prompt Tokens",
        group = "Benchmark",
        argv = "--benchmark-prompt-tokens"
    )]
    #[serde(default)]
    pub benchmark_prompt_tokens: Option<usize>,

    #[job(
        label = "Profile Layers",
        group = "Benchmark",
        argv = "--profile-layers",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub profile_layers: bool,

    #[job(
        label = "Profile Output",
        group = "Benchmark",
        argv = "--profile-output",
        kind = "path"
    )]
    #[serde(default)]
    pub profile_output: Option<String>,

    #[job(label = "KV Quant Bits", group = "Quantization", argv = "--kv-quant")]
    #[serde(default)]
    pub kv_quant: Option<u8>,

    #[job(label = "KV K Bits", group = "Quantization", argv = "--kv-k-bits")]
    #[serde(default)]
    pub kv_k_bits: Option<u8>,

    #[job(label = "KV V Bits", group = "Quantization", argv = "--kv-v-bits")]
    #[serde(default)]
    pub kv_v_bits: Option<u8>,

    #[job(
        label = "KV Group Size",
        group = "Quantization",
        argv = "--kv-group-size",
        default_int = 64
    )]
    #[serde(default = "default_kv_group_size")]
    pub kv_group_size: usize,

    #[job(
        label = "KV TurboQuant",
        group = "Quantization",
        argv = "--kv-turboquant",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub kv_turboquant: bool,

    #[job(label = "TurboQuant Preset", group = "Quantization", argv = "--kv-turboquant-preset", kind = "enum", enum_options = ["q2_5", "q3_5"])]
    #[serde(default)]
    pub kv_turboquant_preset: Option<String>,

    #[job(label = "KV Quant Preset", group = "Quantization", argv = "--kv-quant-preset", kind = "enum", enum_options = ["q2_5", "q3_5"])]
    #[serde(default)]
    pub kv_quant_preset: Option<String>,

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
        label = "KV QJL Correction",
        group = "Quantization",
        argv = "--kv-qjl",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub kv_qjl: bool,

    #[job(
        label = "Detect Repetition",
        group = "Sampling",
        argv = "--detect-repetition",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub detect_repetition: bool,
}

impl Default for InferSpec {
    fn default() -> Self {
        Self {
            model: String::new(),
            lora: None,
            prompt: String::new(),
            max_tokens: default_max_tokens(),
            temperature: None,
            top_k: None,
            top_p: None,
            min_p: None,
            repetition_penalty: None,
            frequency_penalty: None,
            presence_penalty: None,
            seed: None,
            chat: false,
            system: None,
            no_thinking: false,
            hide_thinking: false,
            mode: default_mode(),
            backend: default_backend(),
            draft_model: None,
            compiled: false,
            metal_sampler: false,
            minimal: false,
            tools: None,
            fp8: false,
            experts_dir: None,
            ane: false,
            ane_max_seq_len: default_ane_max_seq_len(),
            ane_real_time: false,
            benchmark: false,
            benchmark_iters: default_benchmark_iters(),
            benchmark_prompt_tokens: None,
            profile_layers: false,
            profile_output: None,
            kv_quant: None,
            kv_k_bits: None,
            kv_v_bits: None,
            kv_group_size: default_kv_group_size(),
            kv_turboquant: false,
            kv_turboquant_preset: None,
            kv_quant_preset: None,
            no_kv_quant: false,
            kv_qjl: false,
            detect_repetition: false,
        }
    }
}

impl InferSpec {
    /// Run descriptor + cross-field validation.
    pub fn normalize(&mut self) -> Result<(), Vec<FieldError>> {
        let mut errs = self.validate_descriptors();
        if self.backend == "dflash" && self.draft_model.is_none() {
            errs.push(FieldError::new(
                "draft_model",
                "required when backend = dflash",
            ));
        }
        if errs.is_empty() { Ok(()) } else { Err(errs) }
    }
}

fn default_max_tokens() -> usize {
    256
}
fn default_mode() -> String {
    "auto".to_string()
}
fn default_backend() -> String {
    "auto".to_string()
}
fn default_ane_max_seq_len() -> usize {
    1024
}
fn default_benchmark_iters() -> usize {
    5
}
fn default_kv_group_size() -> usize {
    64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argv_round_trip() {
        let spec = InferSpec {
            model: "m".into(),
            prompt: "hi".into(),
            ..Default::default()
        };
        let argv = spec.to_argv();
        assert!(argv.contains(&"--model".to_string()));
        assert!(argv.contains(&"--prompt".to_string()));
    }

    #[test]
    fn dflash_requires_draft() {
        let mut spec = InferSpec {
            model: "m".into(),
            prompt: "hi".into(),
            backend: "dflash".into(),
            ..Default::default()
        };
        let res = spec.normalize();
        assert!(res.is_err());
    }
}
