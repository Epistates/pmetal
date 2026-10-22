//! Clap argument struct for `pmetal eval`.

use clap::Args;
use pmetal_core::jobs::EvalSpec;

/// Evaluate a model: perplexity on a dataset, or a benchmark task.
#[derive(Args, Debug)]
pub struct EvalArgs {
    /// Model ID or path
    #[arg(short, long = "model")]
    pub model: String,

    /// Perplexity: the dataset (JSONL file) to score. With --task: a local
    /// file of the task's data in place of the standard dataset, and the
    /// questions of --task custom.
    #[arg(short, long = "dataset")]
    pub dataset: Option<String>,

    /// Score a benchmark instead of perplexity: gsm8k (chain-of-thought math,
    /// exact match), mmlu, arc_challenge and hellaswag (multiple choice by
    /// log-likelihood, accuracy), ifeval-lite (verifiable instructions) or
    /// custom (your JSONL: question and answer, or question and choices).
    #[arg(
        long = "task",
        value_parser = ["gsm8k", "mmlu", "arc_challenge", "hellaswag", "ifeval-lite", "custom"]
    )]
    pub task: Option<String>,

    /// Examples shown before each question. The task's usual number when not
    /// given: gsm8k 5, mmlu 5, arc_challenge 25, hellaswag 10.
    #[arg(long = "fewshot", requires = "task")]
    pub fewshot: Option<usize>,

    /// LoRA adapter path (optional)
    #[arg(long = "lora")]
    pub lora: Option<String>,

    /// Maximum sequence length in tokens (0 = 1024 for perplexity, 8192 for a task)
    #[arg(long = "max-seq-len", default_value = "0")]
    pub max_seq_len: usize,

    /// Number of samples to evaluate (0 = all). A task of one subject scores
    /// its first items; mmlu draws them from every subject with --seed.
    #[arg(long = "num-samples", alias = "limit", default_value = "0")]
    pub num_samples: usize,

    /// Seed of which items a limit keeps and which few-shot examples are drawn.
    #[arg(long = "seed", default_value = "1234")]
    pub seed: u64,

    /// Most sequences scored in one forward pass (a task's log-likelihoods).
    #[arg(long = "batch-size", default_value = "16")]
    pub batch_size: usize,

    /// Most tokens a generation task may produce per item (ifeval-lite asks
    /// for more).
    #[arg(long = "max-gen-tokens", default_value = "256")]
    pub max_gen_tokens: usize,

    /// Write the results as JSON to this file: each metric with its standard
    /// error, per subject where there are any.
    #[arg(long = "output")]
    pub output: Option<String>,

    /// Keep every item's prompt, output and verdict, in a .samples.jsonl file
    /// next to --output.
    #[arg(long = "log-samples", requires = "output")]
    pub log_samples: bool,

    /// Output as JSON
    #[arg(long = "json")]
    pub json: bool,
}

impl From<EvalArgs> for EvalSpec {
    fn from(a: EvalArgs) -> Self {
        EvalSpec {
            model: a.model,
            dataset: a.dataset.unwrap_or_default(),
            task: a.task,
            fewshot: a.fewshot,
            lora: a.lora,
            max_seq_len: a.max_seq_len,
            num_samples: a.num_samples,
            seed: a.seed,
            batch_size: a.batch_size,
            max_gen_tokens: a.max_gen_tokens,
            output: a.output,
            log_samples: a.log_samples,
            json: a.json,
        }
    }
}
