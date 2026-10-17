//! `pmetal train-diffusion` — block-diffusion LoRA fine-tuning of a
//! DiffusionGemma model on `(prompt, response)` pairs.

use crate::{FieldError, JobFields};
use pmetal_core_derive::JobSpec;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, JobSpec)]
#[spec(kind = "TrainDiffusion", subcommand = "train-diffusion")]
#[serde(rename_all = "snake_case")]
pub struct TrainDiffusionSpec {
    #[job(
        label = "Model",
        group = "Model",
        argv = "--model",
        kind = "model_picker",
        help = "A DiffusionGemma checkpoint",
        required
    )]
    #[serde(default)]
    pub model: String,

    #[job(
        label = "Dataset",
        group = "Data",
        argv = "--dataset",
        kind = "dataset_picker",
        help = "JSONL with prompt/response (or context/target, input/output) per line",
        required
    )]
    #[serde(default)]
    pub dataset: String,

    #[job(
        label = "Output Dir",
        group = "Output",
        argv = "--output",
        kind = "path",
        default = "./diffusion-lora-output"
    )]
    #[serde(default = "default_output")]
    pub output: String,

    #[job(
        label = "LoRA r",
        group = "LoRA",
        argv = "--lora-r",
        min = 1,
        max = 1024,
        default_int = 16
    )]
    #[serde(default = "default_lora_r")]
    pub lora_r: usize,

    #[job(
        label = "LoRA α",
        group = "LoRA",
        argv = "--lora-alpha",
        min = 1.0,
        max = 1024.0,
        default_float = 32.0
    )]
    #[serde(default = "default_lora_alpha")]
    pub lora_alpha: f32,

    #[job(
        label = "LoRA Targets (csv)",
        group = "LoRA",
        argv = "--lora-targets",
        csv,
        help = "Projections to adapt in the encoder and decoder; empty adapts every linear layer. MoE routers and experts never are"
    )]
    #[serde(default)]
    pub lora_targets: Vec<String>,

    #[job(
        label = "Quantization",
        group = "LoRA",
        argv = "--quantization",
        kind = "enum",
        enum_options = ["none", "nf4", "fp4", "int8"],
        help = "QLoRA: pack every projection but the routers, and the experts",
        default = "none"
    )]
    #[serde(default = "default_quantization")]
    pub quantization: String,

    #[job(
        label = "Quant Block Size",
        group = "LoRA",
        argv = "--quant-block-size",
        min = 1,
        max = 4096,
        default_int = 64
    )]
    #[serde(default = "default_quant_block_size")]
    pub quant_block_size: usize,

    #[job(
        label = "Double Quant",
        group = "LoRA",
        argv = "--double-quant",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub double_quant: bool,

    #[job(
        label = "Steps",
        group = "Training",
        argv = "--steps",
        help = "Optimizer steps; the dataset is cycled to reach them",
        min = 1,
        default_int = 1000
    )]
    #[serde(default = "default_steps")]
    pub steps: usize,

    #[job(
        label = "Learning Rate",
        group = "Optimization",
        argv = "--learning-rate",
        min = 1e-8,
        max = 1.0,
        default_float = 0.0002
    )]
    #[serde(default = "default_lr")]
    pub learning_rate: f32,

    #[job(
        label = "Weight Decay",
        group = "Optimization",
        argv = "--weight-decay",
        min = 0.0,
        max = 1.0,
        default_float = 0.0
    )]
    #[serde(default)]
    pub weight_decay: f32,

    #[job(
        label = "Max Grad Norm",
        group = "Optimization",
        argv = "--max-grad-norm",
        help = "0 disables clipping",
        min = 0.0,
        max = 1000.0,
        default_float = 1.0
    )]
    #[serde(default = "default_grad_norm")]
    pub max_grad_norm: f32,

    #[job(
        label = "Max Context Len",
        group = "Data",
        argv = "--max-context-len",
        help = "Prompt tokens kept as the encoder context",
        min = 1,
        max = 1_048_576,
        default_int = 512
    )]
    #[serde(default = "default_max_context_len")]
    pub max_context_len: usize,

    #[job(
        label = "Checkpoint Every",
        group = "Output",
        argv = "--checkpoint-every",
        help = "Steps between intermediate adapters; 0 disables them",
        default_int = 0
    )]
    #[serde(default)]
    pub checkpoint_every: usize,

    #[job(
        label = "Log Every",
        group = "Output",
        argv = "--log-every",
        help = "Steps between log lines; 0 disables them",
        default_int = 10
    )]
    #[serde(default = "default_log_every")]
    pub log_every: usize,

    #[job(label = "Seed", group = "Training", argv = "--seed", default_int = 42)]
    #[serde(default = "default_seed")]
    pub seed: u64,
}

impl Default for TrainDiffusionSpec {
    fn default() -> Self {
        Self {
            model: String::new(),
            dataset: String::new(),
            output: default_output(),
            lora_r: default_lora_r(),
            lora_alpha: default_lora_alpha(),
            lora_targets: Vec::new(),
            quantization: default_quantization(),
            quant_block_size: default_quant_block_size(),
            double_quant: false,
            steps: default_steps(),
            learning_rate: default_lr(),
            weight_decay: 0.0,
            max_grad_norm: default_grad_norm(),
            max_context_len: default_max_context_len(),
            checkpoint_every: 0,
            log_every: default_log_every(),
            seed: default_seed(),
        }
    }
}

impl TrainDiffusionSpec {
    pub fn normalize(&mut self) -> Result<(), Vec<FieldError>> {
        let mut errs = self.validate_descriptors();
        if self.double_quant && self.quantization == "none" {
            errs.push(FieldError::new(
                "double_quant",
                "double quantization needs a quantization method",
            ));
        }
        if errs.is_empty() { Ok(()) } else { Err(errs) }
    }
}

fn default_output() -> String {
    "./diffusion-lora-output".to_string()
}
fn default_lora_r() -> usize {
    crate::defaults::LORA_R
}
fn default_lora_alpha() -> f32 {
    crate::defaults::LORA_ALPHA
}
fn default_quantization() -> String {
    "none".to_string()
}
fn default_quant_block_size() -> usize {
    64
}
fn default_steps() -> usize {
    1000
}
fn default_lr() -> f32 {
    2e-4
}
fn default_grad_norm() -> f32 {
    1.0
}
fn default_max_context_len() -> usize {
    512
}
fn default_log_every() -> usize {
    10
}
fn default_seed() -> u64 {
    crate::defaults::SEED
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_adapts_every_linear_layer() {
        let spec = TrainDiffusionSpec {
            model: "m".into(),
            dataset: "d".into(),
            ..Default::default()
        };
        let argv = spec.to_argv();
        assert!(
            !argv.iter().any(|a| a == "--lora-targets"),
            "an empty list is the default and says nothing: {argv:?}"
        );
    }

    #[test]
    fn double_quant_needs_a_quantization_method() {
        let mut spec = TrainDiffusionSpec {
            model: "m".into(),
            dataset: "d".into(),
            double_quant: true,
            ..Default::default()
        };
        assert!(spec.normalize().is_err());
        spec.quantization = "nf4".into();
        spec.normalize().unwrap();
    }

    #[test]
    fn subcommand_and_kind() {
        assert_eq!(TrainDiffusionSpec::subcommand(), "train-diffusion");
        assert_eq!(
            TrainDiffusionSpec::job_kind(),
            crate::JobKind::TrainDiffusion
        );
    }
}
