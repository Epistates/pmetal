//! `pmetal eval` — perplexity on a dataset, or a benchmark task.

use crate::{FieldError, JobFields};
use pmetal_core_derive::JobSpec;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, JobSpec)]
#[spec(kind = "Eval", subcommand = "eval")]
#[serde(rename_all = "snake_case")]
pub struct EvalSpec {
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
        label = "Dataset",
        group = "Data",
        argv = "--dataset",
        kind = "dataset_picker",
        help = "Perplexity: the text to score. With --task: a local file of the task's data to use in place of the standard dataset, and the questions of --task custom"
    )]
    #[serde(default)]
    pub dataset: String,

    #[job(
        label = "Task",
        group = "Task",
        argv = "--task",
        kind = "enum",
        enum_options = ["gsm8k", "mmlu", "arc_challenge", "hellaswag", "ifeval-lite", "custom"],
        help = "Score a benchmark instead of perplexity: gsm8k (chain-of-thought math, exact match), mmlu, arc_challenge and hellaswag (multiple choice by log-likelihood, accuracy), ifeval-lite (verifiable instructions) or custom (your JSONL: question and answer, or question and choices)"
    )]
    #[serde(default)]
    pub task: Option<String>,

    #[job(
        label = "Few-Shot Examples",
        group = "Task",
        argv = "--fewshot",
        min = 0,
        max = 256,
        help = "Examples shown before each question. The task's usual number when not given: gsm8k 5, mmlu 5, arc_challenge 25, hellaswag 10, ifeval-lite and custom 0"
    )]
    #[serde(default)]
    pub fewshot: Option<usize>,

    #[job(
        label = "LoRA Adapter",
        group = "Model",
        argv = "--lora",
        kind = "path"
    )]
    #[serde(default)]
    pub lora: Option<String>,

    #[job(
        label = "Max Seq Len",
        group = "Eval",
        argv = "--max-seq-len",
        help = "0 = 1024 tokens for perplexity, 8192 for a task. A longer sequence loses the start of its context",
        default_int = 0
    )]
    #[serde(default)]
    pub max_seq_len: usize,

    #[job(
        label = "Num Samples",
        group = "Eval",
        argv = "--num-samples",
        help = "Score this many samples (0 = all). A task of one subject scores its first items; mmlu draws them from every subject with --seed",
        default_int = 0
    )]
    #[serde(default)]
    pub num_samples: usize,

    #[job(
        label = "Seed",
        group = "Task",
        argv = "--seed",
        help = "Seeds which items a limit keeps and which few-shot examples are drawn",
        default_int = 1234
    )]
    #[serde(default = "default_seed")]
    pub seed: u64,

    #[job(
        label = "Batch Size",
        group = "Task",
        argv = "--batch-size",
        min = 1,
        max = 1024,
        help = "Most sequences scored in one forward pass (a task's log-likelihoods)",
        default_int = 16
    )]
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,

    #[job(
        label = "Max Generated Tokens",
        group = "Task",
        argv = "--max-gen-tokens",
        min = 1,
        max = 32768,
        help = "Most tokens a generation task may produce per item (ifeval-lite asks for more)",
        default_int = 256
    )]
    #[serde(default = "default_max_gen_tokens")]
    pub max_gen_tokens: usize,

    #[job(
        label = "Results File",
        group = "Output",
        argv = "--output",
        kind = "path",
        help = "Write the results as JSON to this file: each metric with its standard error, per subject where there are any"
    )]
    #[serde(default)]
    pub output: Option<String>,

    #[job(
        label = "Log Samples",
        group = "Output",
        argv = "--log-samples",
        flag,
        default_bool = false,
        help = "Keep every item's prompt, output and verdict, in a .samples.jsonl file next to --output"
    )]
    #[serde(default)]
    pub log_samples: bool,

    #[job(
        label = "JSON Output",
        group = "Output",
        argv = "--json",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub json: bool,
}

impl Default for EvalSpec {
    fn default() -> Self {
        Self {
            model: String::new(),
            dataset: String::new(),
            task: None,
            fewshot: None,
            lora: None,
            max_seq_len: 0,
            num_samples: 0,
            seed: default_seed(),
            batch_size: default_batch_size(),
            max_gen_tokens: default_max_gen_tokens(),
            output: None,
            log_samples: false,
            json: false,
        }
    }
}

impl EvalSpec {
    pub fn normalize(&mut self) -> Result<(), Vec<FieldError>> {
        let mut errs = self.validate_descriptors();
        let has_dataset = !self.dataset.trim().is_empty();
        match self.task.as_deref() {
            None => {
                if !has_dataset {
                    errs.push(FieldError::new(
                        "dataset",
                        "required (or pass --task for a benchmark)",
                    ));
                }
                if self.fewshot.is_some() {
                    errs.push(FieldError::new("fewshot", "needs --task"));
                }
                if self.log_samples {
                    errs.push(FieldError::new("log_samples", "needs --task"));
                }
            }
            Some("custom") => {
                if !has_dataset {
                    errs.push(FieldError::new(
                        "dataset",
                        "required for --task custom: the JSONL file of your questions",
                    ));
                }
            }
            Some(_) => {}
        }
        if self.log_samples && self.output.is_none() {
            errs.push(FieldError::new(
                "log_samples",
                "needs --output, which names the file the samples are written beside",
            ));
        }
        if self.task.as_deref() == Some("ifeval-lite") && self.fewshot.is_some_and(|n| n > 0) {
            errs.push(FieldError::new(
                "fewshot",
                "ifeval-lite takes no few-shot examples",
            ));
        }
        if errs.is_empty() { Ok(()) } else { Err(errs) }
    }
}

fn default_seed() -> u64 {
    1234
}
fn default_batch_size() -> usize {
    16
}
fn default_max_gen_tokens() -> usize {
    256
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fields(mut spec: EvalSpec) -> Vec<String> {
        spec.normalize()
            .err()
            .map(|errs| errs.into_iter().map(|e| e.field).collect())
            .unwrap_or_default()
    }

    #[test]
    fn argv_round_trip() {
        let spec = EvalSpec {
            model: "m".into(),
            dataset: "d".into(),
            ..Default::default()
        };
        let argv = spec.to_argv();
        assert!(argv.contains(&"--model".to_string()));
        assert!(argv.contains(&"--dataset".to_string()));
    }

    #[test]
    fn perplexity_needs_a_dataset_and_a_task_does_not() {
        let base = EvalSpec {
            model: "m".into(),
            ..Default::default()
        };
        assert_eq!(fields(base.clone()), ["dataset"]);
        for task in ["gsm8k", "mmlu", "arc_challenge", "hellaswag", "ifeval-lite"] {
            let spec = EvalSpec {
                task: Some(task.into()),
                ..base.clone()
            };
            assert!(fields(spec).is_empty(), "{task}");
        }
        // Custom is the user's own file.
        let custom = EvalSpec {
            task: Some("custom".into()),
            ..base.clone()
        };
        assert_eq!(fields(custom.clone()), ["dataset"]);
        let with_file = EvalSpec {
            dataset: "q.jsonl".into(),
            ..custom
        };
        assert!(fields(with_file).is_empty());
    }

    #[test]
    fn an_unknown_task_and_options_without_one_are_refused() {
        let base = EvalSpec {
            model: "m".into(),
            dataset: "d".into(),
            ..Default::default()
        };
        let unknown = EvalSpec {
            task: Some("trivia".into()),
            ..base.clone()
        };
        assert_eq!(fields(unknown), ["task"]);
        let stray = EvalSpec {
            fewshot: Some(3),
            ..base.clone()
        };
        assert_eq!(fields(stray), ["fewshot"]);
        let samples_without_a_file = EvalSpec {
            task: Some("gsm8k".into()),
            log_samples: true,
            ..base.clone()
        };
        assert_eq!(fields(samples_without_a_file), ["log_samples"]);
        let ifeval_shots = EvalSpec {
            task: Some("ifeval-lite".into()),
            fewshot: Some(2),
            ..base
        };
        assert_eq!(fields(ifeval_shots), ["fewshot"]);
    }

    #[test]
    fn a_task_is_a_flag_with_its_options_in_the_argv() {
        let spec = EvalSpec {
            model: "m".into(),
            task: Some("mmlu".into()),
            fewshot: Some(0),
            output: Some("out.json".into()),
            log_samples: true,
            ..Default::default()
        };
        let argv = spec.to_argv();
        for flag in ["--task", "--fewshot", "--output", "--log-samples", "--seed"] {
            assert!(argv.contains(&flag.to_string()), "{flag}");
        }
    }
}
