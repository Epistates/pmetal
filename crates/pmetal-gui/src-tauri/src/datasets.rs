//! A look inside a dataset file before a job uses it.

use std::path::Path;

use serde::Serialize;

use crate::error::{AppError, Result};

/// Rows sampled when the caller names no limit.
const DEFAULT_SAMPLE: usize = 100;
/// The most rows a sample may read.
const MAX_SAMPLE: usize = 10_000;

/// The columns of a dataset and a rough idea of how long its rows are, so a
/// form can suggest a sequence length that does not truncate them.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct DatasetPeek {
    pub columns: Vec<String>,
    /// Average tokens a row, estimated at four characters a token.
    pub avg_tokens_estimate: usize,
    pub max_tokens_estimate: usize,
    /// A sequence length that covers 95 of 100 sampled rows, a multiple of 64.
    pub suggested_seq_len: usize,
    pub rows_sampled: usize,
}

/// Peek at the dataset at `path`: a file (JSONL, Parquet, CSV, ...) or a folder
/// of them.
pub fn peek(path: &str, limit: Option<usize>) -> Result<DatasetPeek> {
    let path = Path::new(path.trim());
    if !path.is_absolute() {
        return Err(AppError::new(
            "Give the full path of the dataset, or choose it with the Browse button. \
             A Hub dataset can be looked at once it is downloaded.",
        ));
    }
    if !path.exists() {
        return Err(AppError::new(format!("{} does not exist.", path.display())));
    }
    let sample = limit.unwrap_or(DEFAULT_SAMPLE).clamp(1, MAX_SAMPLE);

    let columns = pmetal::data::peek_columns(path).map_err(|e| {
        AppError::new(format!(
            "Could not read the columns of {}: {e}. Is it a JSONL, JSON, CSV or Parquet dataset?",
            path.display()
        ))
    })?;
    let chars = pmetal::data::sample_row_chars(path, sample).unwrap_or_default();
    let mut tokens: Vec<usize> = chars.iter().map(|c| c / 4).collect();
    let avg = if tokens.is_empty() {
        0
    } else {
        tokens.iter().sum::<usize>() / tokens.len()
    };
    let max = tokens.iter().copied().max().unwrap_or(0);
    tokens.sort_unstable();
    let p95 = tokens
        .get(tokens.len() * 95 / 100)
        .or(tokens.last())
        .copied()
        .unwrap_or(0);
    Ok(DatasetPeek {
        columns,
        avg_tokens_estimate: avg,
        max_tokens_estimate: max,
        suggested_seq_len: if p95 > 0 { p95.div_ceil(64) * 64 } else { 2048 },
        rows_sampled: tokens.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_jsonl_file_gives_its_columns_and_a_sequence_length_that_fits_its_rows() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("d.jsonl");
        let long = "x".repeat(4 * 300);
        let lines: Vec<String> = (0..20)
            .map(|i| {
                let text = if i % 2 == 0 { long.as_str() } else { "short" };
                format!(r#"{{"text":"{text}","label":{i}}}"#)
            })
            .collect();
        std::fs::write(&file, lines.join("\n")).unwrap();

        let peek = peek(file.to_str().unwrap(), None).unwrap();

        assert!(peek.columns.contains(&"text".to_string()), "{peek:?}");
        assert_eq!(peek.rows_sampled, 20);
        assert!(peek.max_tokens_estimate >= 300, "{peek:?}");
        assert_eq!(peek.suggested_seq_len % 64, 0);
        assert!(peek.suggested_seq_len >= peek.max_tokens_estimate / 2);
    }

    #[test]
    fn a_path_that_is_not_a_dataset_says_so() {
        assert!(
            peek("relative.jsonl", None)
                .unwrap_err()
                .message()
                .contains("full path")
        );
        assert!(
            peek("/no/such/file.jsonl", None)
                .unwrap_err()
                .message()
                .contains("does not exist")
        );
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("d.bin");
        std::fs::write(&file, [0u8, 159, 146, 150]).unwrap();
        assert!(peek(file.to_str().unwrap(), None).is_err());
    }
}
