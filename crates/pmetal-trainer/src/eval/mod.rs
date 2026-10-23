//! Task-based evaluation of a model: GSM8K, MMLU, ARC, HellaSwag, a subset of
//! IFEval, and tasks of your own.
//!
//! A task is how a benchmark turns a dataset row into a prompt, what it asks
//! of the model, and how it scores the answer. These follow the widely used
//! definitions, so a number here means what the same benchmark's number means
//! elsewhere:
//!
//! | Task | Asks | Prompt | Scored |
//! |---|---|---|---|
//! | `gsm8k` | a greedy generation, stopped at the next `Question:` | `Question: {question}\nAnswer:` after the few-shot examples' chain-of-thought answers | exact match of the number after `#### ` (strict) and of the last number (flexible) |
//! | `mmlu` | the log-likelihood of ` A`, ` B`, ` C`, ` D` | the subject's description, the few-shot examples of that subject, then the question and its lettered choices, ending `Answer:` | accuracy |
//! | `arc_challenge` | the log-likelihood of each choice's text | `Question: {question}\nAnswer:` | accuracy, and accuracy with the log-likelihood normalised by the choice's length in characters |
//! | `hellaswag` | the log-likelihood of each ending | the activity label and context, with the dataset's markup cleaned out | accuracy and normalised accuracy |
//! | `ifeval-lite` | a greedy generation | the prompt as it is | whether each verifiable instruction is followed, strictly and loosely, per prompt and per instruction |
//! | `custom` | a generation from questions with answers, or the log-likelihood of choices | `Question: {question}\nAnswer:` | exact match, or accuracy |
//!
//! Each reports its metrics with their standard errors ([`Metric`]), and a
//! run can keep every item's prompt, output and verdict ([`TaskReport::samples_log`]).
//! The model is anything that can score a continuation and generate
//! ([`EvalModel`]); [`LoadedModel`] is the inference path's own.
//!
//! What is not the same as elsewhere, because it cannot be: the few-shot
//! examples of a task that draws them at random are drawn from this
//! implementation's seeded generator, not Python's, so the same `seed` gives
//! the same examples run to run, but not the ones another implementation drew. MMLU's
//! come from the subject's own `dev` split, in order, as they are everywhere.

mod datasets;
mod gsm8k;
mod harness;
mod ifeval;
mod model;
mod stats;
mod tasks;

pub use datasets::{TaskData, TaskSource, task_source};
pub use harness::{encode_pair, evaluate};
pub use model::{EvalModel, LlRequest, LlResult, LoadedModel};
pub use stats::{Metric, mean, mean_stderr};

#[cfg(feature = "hub")]
pub use datasets::download_task_data;

use std::collections::BTreeMap;
use std::fmt;
use std::str::FromStr;

use serde::Serialize;
use serde_json::Value;

/// Why an evaluation could not run.
#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    /// The task's data is missing, unreadable or not the task's layout.
    #[error("{0}")]
    Data(String),
    /// The options do not make sense for the task.
    #[error("{0}")]
    Config(String),
    /// The model failed.
    #[error("{0}")]
    Model(String),
}

/// A benchmark.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Task {
    /// Grade-school math word problems, answered with a chain of thought.
    Gsm8k,
    /// 57 subjects of multiple-choice questions.
    Mmlu,
    /// The AI2 Reasoning Challenge's hard set.
    ArcChallenge,
    /// Choosing a sentence's plausible ending.
    Hellaswag,
    /// Following verifiable instructions, the subset that needs no language tools.
    IfevalLite,
    /// Questions of the user's own, from a JSONL file.
    Custom,
}

impl Task {
    /// Every task.
    pub const ALL: [Task; 6] = [
        Task::Gsm8k,
        Task::Mmlu,
        Task::ArcChallenge,
        Task::Hellaswag,
        Task::IfevalLite,
        Task::Custom,
    ];

    /// The task's name, as `--task` spells it.
    pub fn name(self) -> &'static str {
        match self {
            Task::Gsm8k => "gsm8k",
            Task::Mmlu => "mmlu",
            Task::ArcChallenge => "arc_challenge",
            Task::Hellaswag => "hellaswag",
            Task::IfevalLite => "ifeval-lite",
            Task::Custom => "custom",
        }
    }

    /// The number of few-shot examples the task is usually reported with.
    pub fn default_fewshot(self) -> usize {
        match self {
            Task::Gsm8k | Task::Mmlu => 5,
            Task::ArcChallenge => 25,
            Task::Hellaswag => 10,
            Task::IfevalLite | Task::Custom => 0,
        }
    }

    /// Whether the task is scored on generated text rather than on
    /// log-likelihoods.
    pub fn generates(self) -> bool {
        matches!(self, Task::Gsm8k | Task::IfevalLite)
    }
}

impl fmt::Display for Task {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for Task {
    type Err = String;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        let normalized = name.trim().to_ascii_lowercase().replace('_', "-");
        Task::ALL
            .into_iter()
            .find(|task| task.name().replace('_', "-") == normalized)
            .ok_or_else(|| {
                format!(
                    "unknown task '{name}'; valid: {}",
                    Task::ALL.map(Task::name).join(", ")
                )
            })
    }
}

/// How a task is run.
#[derive(Debug, Clone)]
pub struct TaskOptions {
    /// Few-shot examples per item; `None` is the task's usual number
    /// ([`Task::default_fewshot`]).
    pub fewshot: Option<usize>,
    /// Score only this many items; 0 scores them all. A task of one subject
    /// scores its first items, as the benchmarks do; a task of many subjects
    /// (MMLU) draws them from all of them with `seed`, since the first items of
    /// the data are one subject.
    pub limit: usize,
    /// The seed of everything random: which items a limit keeps and which
    /// few-shot examples a task draws.
    pub seed: u64,
    /// Most tokens a generation may take (a task may ask for more: IFEval's
    /// answers are long).
    pub max_gen_tokens: usize,
    /// Keep every item's prompt, output and verdict in the report.
    pub log_samples: bool,
}

impl Default for TaskOptions {
    fn default() -> Self {
        Self {
            fewshot: None,
            limit: 0,
            seed: 1234,
            max_gen_tokens: 256,
            log_samples: false,
        }
    }
}

/// The score of one subject of a task.
#[derive(Debug, Clone, Serialize)]
pub struct GroupReport {
    /// Items scored.
    pub samples: usize,
    /// The subject's metrics.
    pub metrics: BTreeMap<String, Metric>,
}

/// What a task run measured.
#[derive(Debug, Clone, Serialize)]
pub struct TaskReport {
    /// The task.
    pub task: String,
    /// Few-shot examples per item.
    pub fewshot: usize,
    /// Items scored.
    pub samples: usize,
    /// Items the task skipped (an instruction no checker is written for).
    pub skipped: usize,
    /// The task's metrics, by name, with their standard errors.
    pub metrics: BTreeMap<String, Metric>,
    /// Per-subject metrics, for a task with subjects.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub groups: BTreeMap<String, GroupReport>,
    /// Every item's prompt, output and verdict, when
    /// [`TaskOptions::log_samples`] asked for them.
    #[serde(skip)]
    pub samples_log: Vec<Value>,
}

impl TaskReport {
    /// One metric's value.
    pub fn value(&self, metric: &str) -> Option<f64> {
        self.metrics.get(metric).map(|m| m.value)
    }
}
