//! Where each task's data is, and reading it.
//!
//! The standard tasks name Hugging Face datasets and the files of them they
//! use; [`download_task_data`] fetches exactly those files (not the dataset
//! repository, which for MMLU holds every subject twice over and a hundred
//! megabytes of auxiliary training data). Any task can be given a local file
//! instead, for an offline run or a variant of the data.

use std::path::{Path, PathBuf};

use serde_json::Value;

use super::{EvalError, Task};
use crate::preference::data::read_rows;

/// A task's data files on the Hub.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TaskSource {
    /// The dataset repository.
    pub dataset: &'static str,
    /// The file the task is scored on.
    pub test: &'static str,
    /// The file its few-shot examples come from, for a task that uses any.
    pub fewshot: Option<&'static str>,
}

/// The Hub files of a standard task; `None` for `custom`, which has only the
/// user's.
pub fn task_source(task: Task) -> Option<TaskSource> {
    Some(match task {
        Task::Gsm8k => TaskSource {
            dataset: "openai/gsm8k",
            test: "main/test-00000-of-00001.parquet",
            fewshot: Some("main/train-00000-of-00001.parquet"),
        },
        Task::Mmlu => TaskSource {
            dataset: "cais/mmlu",
            test: "all/test-00000-of-00001.parquet",
            fewshot: Some("all/dev-00000-of-00001.parquet"),
        },
        Task::ArcChallenge => TaskSource {
            dataset: "allenai/ai2_arc",
            test: "ARC-Challenge/test-00000-of-00001.parquet",
            fewshot: Some("ARC-Challenge/train-00000-of-00001.parquet"),
        },
        Task::Hellaswag => TaskSource {
            dataset: "Rowan/hellaswag",
            test: "data/validation-00000-of-00001.parquet",
            fewshot: Some("data/train-00000-of-00001.parquet"),
        },
        Task::IfevalLite => TaskSource {
            dataset: "google/IFEval",
            test: "ifeval_input_data.jsonl",
            fewshot: None,
        },
        Task::Custom => return None,
    })
}

/// A task's rows: the ones it is scored on, and the pool its few-shot examples
/// are drawn from.
#[derive(Debug, Clone, Default)]
pub struct TaskData {
    /// The rows to score.
    pub test: Vec<Value>,
    /// The rows the few-shot examples come from. Empty when the task is run
    /// without any, or draws them from `test`.
    pub fewshot: Vec<Value>,
}

impl TaskData {
    /// The rows of `test`, and of `fewshot` if there is one.
    ///
    /// # Errors
    ///
    /// [`EvalError::Data`] if a file cannot be read or has no rows.
    pub fn from_files(test: &Path, fewshot: Option<&Path>) -> Result<Self, EvalError> {
        let read = |path: &Path| {
            read_rows(path)
                .map_err(|e| EvalError::Data(format!("cannot read {}: {e}", path.display())))
        };
        let test_rows = read(test)?;
        if test_rows.is_empty() {
            return Err(EvalError::Data(format!("{} has no rows", test.display())));
        }
        Ok(Self {
            test: test_rows,
            fewshot: match fewshot {
                Some(path) => read(path)?,
                None => Vec::new(),
            },
        })
    }
}

/// Download the files of `task` (the few-shot file only when `fewshot` examples
/// are wanted) and read them.
///
/// # Errors
///
/// [`EvalError::Data`] for a download or read that fails, and for `custom`,
/// which has no dataset on the Hub.
#[cfg(feature = "hub")]
pub async fn download_task_data(task: Task, fewshot: usize) -> Result<TaskData, EvalError> {
    let source = task_source(task).ok_or_else(|| {
        EvalError::Config("`custom` evaluates a file of your own: pass --dataset".into())
    })?;
    let fetch = |file: &'static str| async move {
        tracing::info!("Fetching {}/{}", source.dataset, file);
        pmetal_hub::download_dataset_file(source.dataset, file, None, None)
            .await
            .map_err(|e| EvalError::Data(format!("cannot download {}/{file}: {e}", source.dataset)))
    };
    let test = fetch(source.test).await?;
    let pool: Option<PathBuf> = match source.fewshot {
        Some(file) if fewshot > 0 => Some(fetch(file).await?),
        _ => None,
    };
    TaskData::from_files(&test, pool.as_deref())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_standard_task_names_its_files_and_custom_has_none() {
        for task in Task::ALL {
            match (task, task_source(task)) {
                (Task::Custom, source) => assert!(source.is_none()),
                (_, Some(source)) => {
                    assert!(source.dataset.contains('/'), "{task}");
                    assert!(source.test.ends_with(".parquet") || source.test.ends_with(".jsonl"));
                    // The few-shot tasks are the ones with a default number.
                    assert_eq!(
                        source.fewshot.is_some(),
                        task.default_fewshot() > 0,
                        "{task}"
                    );
                }
                (_, None) => panic!("{task} has no source"),
            }
        }
    }

    #[test]
    fn rows_are_read_from_files_and_an_empty_one_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let test = dir.path().join("test.jsonl");
        let pool = dir.path().join("pool.jsonl");
        std::fs::write(&test, "{\"question\": \"a\"}\n{\"question\": \"b\"}\n").unwrap();
        std::fs::write(&pool, "{\"question\": \"c\"}\n").unwrap();
        let data = TaskData::from_files(&test, Some(&pool)).unwrap();
        assert_eq!((data.test.len(), data.fewshot.len()), (2, 1));
        assert_eq!(data.test[1]["question"], "b");

        std::fs::write(&test, "").unwrap();
        assert!(matches!(
            TaskData::from_files(&test, None),
            Err(EvalError::Data(m)) if m.contains("no rows")
        ));
        assert!(TaskData::from_files(&dir.path().join("missing.jsonl"), None).is_err());
    }
}
