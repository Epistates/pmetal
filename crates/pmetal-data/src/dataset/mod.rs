//! Dataset types and loading.
//!
//! A dataset is read in three steps, each in its own module:
//!
//! 1. [`crate::rows`] streams the file's rows (JSONL, JSON, Parquet, Arrow,
//!    CSV; a directory is its train split's shards);
//! 2. `format` reads a row as a document, a prompt and completion, or a
//!    conversation, by the [`DatasetFormat`] named or detected from its
//!    columns;
//! 3. `render` turns that into text and tokens, with the loss mask decided
//!    in bytes of the text and carried to the tokens by their offsets.
//!
//! [`TrainingDataset::load`] runs all three over bounded chunks of rows, the
//! tokenization in parallel.

mod embedding;
mod format;
mod mix;
mod render;

pub use embedding::{EmbeddingDataset, EmbeddingPair, EmbeddingTriplet};
pub use format::{DatasetColumnConfig, DatasetFormat};
pub use mix::MixSource;
pub use render::{AssistantLoss, TextSample};

use super::chat_templates::ChatTemplate;
use crate::dataloader::shuffled_indices;
use crate::rows::{Row, RowReader};
use pmetal_core::{PMetalError, Result};
use rayon::prelude::*;
use std::path::{Path, PathBuf};

/// Label of a position the loss ignores.
pub const IGNORE_INDEX: i64 = -100;

/// Rows read, rendered and tokenized at a time.
const CHUNK_ROWS: usize = 2048;

/// A training sample.
///
/// Every token attends: a sample carries no attention mask of its own (a
/// stored mask of ones cost four bytes a token, as much again as the ids),
/// and the loaders build the batch's mask from the lengths.
#[derive(Debug, Clone)]
pub struct Sample {
    /// Input token IDs.
    pub input_ids: Vec<u32>,
    /// Labels (for supervised training). Use -100 for tokens to ignore in loss.
    pub labels: Option<Vec<i64>>,
    /// Image paths (for multimodal training).
    pub images: Option<Vec<PathBuf>>,
}

impl Sample {
    /// Create a new sample with input_ids only.
    pub fn new(input_ids: Vec<u32>) -> Self {
        Self {
            input_ids,
            labels: None,
            images: None,
        }
    }

    /// Create a sample with input_ids and labels.
    pub fn with_labels(input_ids: Vec<u32>, labels: Vec<i64>) -> Self {
        Self {
            input_ids,
            labels: Some(labels),
            images: None,
        }
    }

    /// Create a sample with input_ids, labels, and images.
    pub fn with_images(input_ids: Vec<u32>, labels: Vec<i64>, images: Vec<PathBuf>) -> Self {
        Self {
            input_ids,
            labels: Some(labels),
            images: Some(images),
        }
    }
}

/// How to load a dataset: [`TrainingDataset::load`].
#[derive(Debug, Clone)]
pub struct DatasetLoadOptions<'a> {
    /// The rows' format; [`DatasetFormat::Auto`] detects it from the first row.
    pub format: DatasetFormat,
    /// Longest sample, in tokens; longer ones are truncated.
    pub max_length: usize,
    /// The model's chat template, for conversations. Without one a
    /// conversation is rendered as plain `Role: content` blocks.
    pub template: Option<&'a ChatTemplate>,
    /// Custom columns; naming any overrides `format`.
    pub columns: Option<&'a DatasetColumnConfig>,
    /// Which assistant turns carry loss.
    pub assistant_loss: AssistantLoss,
}

impl<'a> DatasetLoadOptions<'a> {
    /// Detect the format, train on every assistant turn, truncate to
    /// `max_length` tokens.
    pub fn new(max_length: usize) -> Self {
        Self {
            format: DatasetFormat::Auto,
            max_length,
            template: None,
            columns: None,
            assistant_loss: AssistantLoss::default(),
        }
    }

    /// Read the rows as `format`.
    pub fn format(mut self, format: DatasetFormat) -> Self {
        self.format = format;
        self
    }

    /// Render conversations with `template`.
    pub fn template(mut self, template: Option<&'a ChatTemplate>) -> Self {
        self.template = template;
        self
    }

    /// Take the text from custom columns.
    pub fn columns(mut self, columns: Option<&'a DatasetColumnConfig>) -> Self {
        self.columns = columns;
        self
    }

    /// Choose which assistant turns carry loss.
    pub fn assistant_loss(mut self, loss: AssistantLoss) -> Self {
        self.assistant_loss = loss;
        self
    }
}

fn invalid_data(message: impl Into<String>) -> PMetalError {
    PMetalError::Io(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        message.into(),
    ))
}

/// Read `path` a chunk of rows at a time, parse each chunk in parallel and
/// hand `consume` the parsed rows. `Auto` is resolved on the first row.
fn for_each_chunk<T: Send>(
    path: &Path,
    format: &DatasetFormat,
    columns: Option<&DatasetColumnConfig>,
    parse: impl Fn(&DatasetFormat, &Row) -> Result<T> + Sync,
    mut consume: impl FnMut(T) -> Result<()>,
) -> Result<()> {
    let mut reader = RowReader::open(path)?;
    let format = format::effective_format(format.clone(), columns);
    let mut resolved: Option<DatasetFormat> = None;
    let mut row_number = 0usize;
    loop {
        let mut chunk: Vec<(usize, Row)> = Vec::with_capacity(CHUNK_ROWS);
        for row in reader.by_ref().take(CHUNK_ROWS) {
            row_number += 1;
            chunk.push((row_number, row?));
        }
        if chunk.is_empty() {
            return Ok(());
        }
        if resolved.is_none() {
            let detected = match &format {
                DatasetFormat::Auto => DatasetFormat::detect(&chunk[0].1)
                    .map_err(|e| invalid_data(format!("row 1: {e}")))?,
                other => other.clone(),
            };
            if matches!(format, DatasetFormat::Auto) {
                tracing::info!("Detected dataset format: {detected:?}");
            }
            resolved = Some(detected);
        }
        let format = resolved.as_ref().expect("resolved above");
        let parsed: Vec<Result<T>> = chunk
            .par_iter()
            .map(|(number, row)| {
                parse(format, row).map_err(|e| invalid_data(format!("row {number}: {e}")))
            })
            .collect();
        for item in parsed {
            consume(item?)?;
        }
    }
}

impl DatasetFormat {
    /// [`render_row`](Self::render_row) for putting a dataset in a model's chat
    /// format: a row that is a question and its answer (Alpaca, reasoning,
    /// prompt and completion of text) is rendered as the conversation it
    /// stands for, where training reads it as a completion in its own layout.
    ///
    /// # Errors
    /// As [`render_row`](Self::render_row).
    pub fn render_row_as_chat(
        &self,
        row: &Row,
        template: Option<&ChatTemplate>,
    ) -> Result<Vec<TextSample>> {
        Ok(render::render(
            self.parse_as_chat(row)?,
            template,
            AssistantLoss::default(),
        ))
    }

    /// Read one row in this format and render it to text, with the chat
    /// template for a conversation (or plain `Role: content` blocks without
    /// one). Usually one sample; see [`TextSample`].
    ///
    /// `Auto` is resolved against this row. A caller that reads many rows
    /// and wants them all read as the first one is resolves it once with
    /// [`DatasetFormat::detect`].
    ///
    /// # Errors
    /// A row without the columns of the format; the message lists the row's
    /// columns.
    pub fn render_row(
        &self,
        row: &Row,
        template: Option<&ChatTemplate>,
    ) -> Result<Vec<TextSample>> {
        Ok(render::render(
            self.parse(row)?,
            template,
            AssistantLoss::default(),
        ))
    }
}

/// Dataset for training.
#[derive(Clone)]
pub struct TrainingDataset {
    samples: Vec<Sample>,
    /// Original token lengths before truncation, recorded when loading from
    /// a file. Used by `compute_statistics` for accurate truncation reporting.
    /// Empty when the dataset is constructed via `from_samples` or other paths that
    /// don't track pre-truncation lengths.
    original_lengths: Vec<usize>,
}

impl TrainingDataset {
    /// Create a new empty dataset.
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
            original_lengths: Vec::new(),
        }
    }

    /// Create a dataset from pre-tokenized samples.
    pub fn from_samples(samples: Vec<Sample>) -> Self {
        Self {
            samples,
            original_lengths: Vec::new(),
        }
    }

    /// Load and tokenize the dataset at `path`: a file in any format
    /// [`crate::rows`] reads, or a directory holding one (its train split,
    /// every shard).
    ///
    /// Rows are read, rendered and tokenized in bounded chunks, so memory
    /// holds the tokens of the dataset and no more than a chunk of its text.
    /// A sample whose trained tokens were all cut off by `max_length` (a
    /// prompt as long as the context) is dropped, with a warning: it would
    /// reach the loss as a row of ignored labels.
    ///
    /// # Errors
    /// A row that cannot be read, or that has none of the columns of the
    /// format, fails the load, naming the file, the row and the columns.
    pub fn load(
        path: impl AsRef<Path>,
        tokenizer: &super::Tokenizer,
        options: &DatasetLoadOptions<'_>,
    ) -> Result<Self> {
        let mut samples = Vec::new();
        let mut original_lengths = Vec::new();
        let mut dropped = 0usize;
        for_each_chunk(
            path.as_ref(),
            &options.format,
            options.columns,
            |format, row| {
                let example = format.parse(row)?;
                render::render(example, options.template, options.assistant_loss)
                    .iter()
                    .map(|text| render::tokenize(tokenizer, text, options.max_length))
                    .collect::<Result<Vec<_>>>()
            },
            |row_samples| {
                for tokenized in row_samples {
                    if !tokenized.has_trained_tokens() {
                        dropped += 1;
                        continue;
                    }
                    original_lengths.push(tokenized.original_len);
                    samples.push(Sample {
                        input_ids: tokenized.input_ids,
                        labels: Some(tokenized.labels),
                        images: None,
                    });
                }
                Ok(())
            },
        )?;
        if dropped > 0 {
            tracing::warn!(
                "Dropped {dropped} sample(s) with no trained token left after truncation to \
                 {} tokens (their prompt fills the context)",
                options.max_length
            );
        }
        Ok(Self {
            samples,
            original_lengths,
        })
    }

    /// Load and tokenize a dataset file (any format [`TrainingDataset::load`]
    /// reads; the name is from when JSONL was the only one).
    ///
    /// # Arguments
    /// * `path` - Path to the dataset file or directory
    /// * `tokenizer` - Tokenizer to use
    /// * `format` - Dataset format (or Auto to detect)
    /// * `max_length` - Maximum sequence length
    /// * `template` - Optional chat template for formatting conversations
    /// * `columns` - Optional custom column config; when it names a column other
    ///   than the default `text`, it overrides format detection.
    pub fn from_jsonl_tokenized<P: AsRef<Path>>(
        path: P,
        tokenizer: &super::Tokenizer,
        format: DatasetFormat,
        max_length: usize,
        template: Option<&ChatTemplate>,
        columns: Option<&DatasetColumnConfig>,
    ) -> Result<Self> {
        Self::load(
            path,
            tokenizer,
            &DatasetLoadOptions::new(max_length)
                .format(format)
                .template(template)
                .columns(columns),
        )
    }

    /// Load and tokenize a dataset file, taking the text from `text_column`
    /// and, for SFT label masking, the prompt from `prompt_column`.
    pub fn from_parquet_tokenized<P: AsRef<Path>>(
        path: P,
        tokenizer: &super::Tokenizer,
        text_column: &str,
        max_length: usize,
        prompt_column: Option<&str>,
    ) -> Result<Self> {
        let columns = DatasetColumnConfig {
            text_column: Some(text_column.to_string()),
            prompt_column: prompt_column.map(str::to_string),
            ..Default::default()
        };
        Self::from_jsonl_tokenized(
            path,
            tokenizer,
            DatasetFormat::Auto,
            max_length,
            None,
            Some(&columns),
        )
    }

    /// Render a dataset's rows to text without tokenizing them.
    ///
    /// Reads any format [`TrainingDataset::load`] does; `path` may be a
    /// directory. Conversations are rendered with `template`, or as plain
    /// `Role: content` blocks without one.
    pub fn load_jsonl_text<P: AsRef<Path>>(
        path: P,
        format: DatasetFormat,
        template: Option<&ChatTemplate>,
    ) -> Result<Vec<TextSample>> {
        Self::load_text(path, format, template, None)
    }

    /// Render a dataset file's rows to text, taking the text from
    /// `text_column` and the prompt from `prompt_column`.
    pub fn load_parquet_text<P: AsRef<Path>>(
        path: P,
        text_column: &str,
        prompt_column: Option<&str>,
    ) -> Result<Vec<TextSample>> {
        let columns = DatasetColumnConfig {
            text_column: Some(text_column.to_string()),
            prompt_column: prompt_column.map(str::to_string),
            ..Default::default()
        };
        Self::load_text(path, DatasetFormat::Auto, None, Some(&columns))
    }

    fn load_text(
        path: impl AsRef<Path>,
        format: DatasetFormat,
        template: Option<&ChatTemplate>,
        columns: Option<&DatasetColumnConfig>,
    ) -> Result<Vec<TextSample>> {
        let mut out = Vec::new();
        for_each_chunk(
            path.as_ref(),
            &format,
            columns,
            |format, row| format.render_row(row, template),
            |row_samples| {
                out.extend(row_samples);
                Ok(())
            },
        )?;
        Ok(out)
    }

    /// The path to hand a loader for the dataset at `path`: the file itself
    /// when the dataset is one file, and `path` (a directory) when it is
    /// several (the shards of a split), which the loaders read whole.
    pub fn resolve_dataset_path_pub(path: &Path) -> Result<PathBuf> {
        let files = crate::rows::resolve_files(path)?;
        Ok(match files.as_slice() {
            [only] => only.clone(),
            _ => path.to_path_buf(),
        })
    }

    /// Get the number of samples.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Get a sample by index.
    pub fn get(&self, index: usize) -> Option<&Sample> {
        self.samples.get(index)
    }

    /// Get all samples.
    pub fn samples(&self) -> &[Sample] {
        &self.samples
    }

    /// Shuffle the dataset; the order depends on `seed` alone.
    pub fn shuffle(&mut self, seed: u64) {
        let order = shuffled_indices(self.samples.len(), seed);
        self.samples = permute(std::mem::take(&mut self.samples), &order);
        if self.original_lengths.len() == order.len() {
            self.original_lengths = permute(std::mem::take(&mut self.original_lengths), &order);
        }
    }

    /// Split the dataset into train and validation sets.
    ///
    /// `val_ratio` is the share of the (shuffled) samples that go to
    /// validation, clamped to `0.0..=1.0`.
    pub fn train_val_split(mut self, val_ratio: f32, seed: u64) -> (Self, Self) {
        self.shuffle(seed);
        let n = self.samples.len();
        let val_size = ((n as f32 * val_ratio.clamp(0.0, 1.0)).round() as usize).min(n);
        let val_orig = if self.original_lengths.len() == n {
            self.original_lengths.split_off(n - val_size)
        } else {
            Vec::new()
        };
        let val_samples = self.samples.split_off(n - val_size);

        (
            self,
            Self {
                samples: val_samples,
                original_lengths: val_orig,
            },
        )
    }

    /// Compute sequence-length statistics over this dataset.
    ///
    /// Uses `original_lengths` (pre-truncation) when available; falls back to
    /// the post-truncation lengths stored in `samples`.
    ///
    /// `max_seq_len` is used to determine which samples were truncated and to
    /// compute `suggested_max_seq_len`.
    pub fn compute_statistics(&self, max_seq_len: usize) -> DatasetStatistics {
        let post_lengths: Vec<usize> = self.samples.iter().map(|s| s.input_ids.len()).collect();

        // Use pre-truncation lengths for distribution stats when available.
        let orig: &[usize] = if self.original_lengths.len() == self.samples.len()
            && !self.original_lengths.is_empty()
        {
            &self.original_lengths
        } else {
            &post_lengths
        };

        if orig.is_empty() {
            return DatasetStatistics {
                total_samples: 0,
                min_length: 0,
                max_length: 0,
                mean_length: 0.0,
                median_length: 0,
                p95_length: 0,
                p99_length: 0,
                truncated_count: 0,
                truncated_pct: 0.0,
                suggested_max_seq_len: max_seq_len,
                columns: Vec::new(),
            };
        }

        let total = orig.len();
        let mut sorted = orig.to_vec();
        sorted.sort_unstable();

        let mean = orig.iter().sum::<usize>() as f64 / total as f64;
        let median = sorted[total / 2];
        let p95 = sorted[((total as f64 * 0.95) as usize).min(total - 1)];
        let p99 = sorted[((total as f64 * 0.99) as usize).min(total - 1)];

        let truncated = orig.iter().filter(|&&l| l > max_seq_len).count();

        // Suggest the next multiple of 64 at or above p95 (GPU-friendly alignment),
        // capped at max_seq_len. Guard against p95==0.
        let suggested = if p95 > 0 {
            (p95.div_ceil(64) * 64).min(max_seq_len)
        } else {
            max_seq_len
        };

        DatasetStatistics {
            total_samples: total,
            min_length: *sorted.first().unwrap(),
            max_length: *sorted.last().unwrap(),
            mean_length: mean,
            median_length: median,
            p95_length: p95,
            p99_length: p99,
            truncated_count: truncated,
            truncated_pct: truncated as f64 / total as f64 * 100.0,
            suggested_max_seq_len: suggested,
            columns: Vec::new(),
        }
    }

    /// Validate sequence length settings and return human-readable warnings.
    ///
    /// Issues a warning when more than 10% of samples are truncated, and a note
    /// when the average length is much shorter than `max_seq_len` (suggesting
    /// that a smaller value would give faster training).
    pub fn validate_seq_len(&self, max_seq_len: usize) -> Vec<String> {
        let stats = self.compute_statistics(max_seq_len);
        let mut warnings = Vec::new();

        if stats.truncated_pct > 50.0 {
            warnings.push(format!(
                "WARNING: {:.0}% of samples truncated at max_seq_len={}. \
                 Consider increasing to {} (dataset p95).",
                stats.truncated_pct, max_seq_len, stats.p95_length
            ));
        } else if stats.truncated_pct > 10.0 {
            warnings.push(format!(
                "NOTE: {:.0}% of samples truncated at max_seq_len={}. \
                 Dataset p95 length is {}.",
                stats.truncated_pct, max_seq_len, stats.p95_length
            ));
        }

        if stats.mean_length < max_seq_len as f64 * 0.25 {
            warnings.push(format!(
                "NOTE: Average sample length ({:.0}) is much shorter than max_seq_len ({}). \
                 Consider reducing max_seq_len for faster training.",
                stats.mean_length, max_seq_len
            ));
        }

        warnings
    }

    /// Suggest an optimal `max_seq_len` based on actual data lengths.
    ///
    /// Returns a power-of-two value that covers p99 of the data without
    /// excessive waste. Useful for display/recommendation but NOT applied
    /// automatically — users should have full control over this setting.
    pub fn suggested_seq_len(&self, configured: usize) -> usize {
        let stats = self.compute_statistics(configured);
        stats.suggested_max_seq_len
    }
}

impl Default for TrainingDataset {
    fn default() -> Self {
        Self::new()
    }
}

impl IntoIterator for TrainingDataset {
    type Item = Sample;
    type IntoIter = std::vec::IntoIter<Sample>;

    fn into_iter(self) -> Self::IntoIter {
        self.samples.into_iter()
    }
}

impl<'a> IntoIterator for &'a TrainingDataset {
    type Item = &'a Sample;
    type IntoIter = std::slice::Iter<'a, Sample>;

    fn into_iter(self) -> Self::IntoIter {
        self.samples.iter()
    }
}

/// Statistics computed over a tokenized `TrainingDataset`.
///
/// Includes sequence-length distribution metrics and truncation diagnostics.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DatasetStatistics {
    /// Total number of samples in the dataset.
    pub total_samples: usize,
    /// Minimum token count across all samples (post-truncation).
    pub min_length: usize,
    /// Maximum token count across all samples (post-truncation).
    pub max_length: usize,
    /// Mean token count across all samples (post-truncation).
    pub mean_length: f64,
    /// Median token count (post-truncation).
    pub median_length: usize,
    /// 95th percentile token count (pre-truncation original lengths).
    pub p95_length: usize,
    /// 99th percentile token count (pre-truncation original lengths).
    pub p99_length: usize,
    /// Number of samples whose original length exceeded `max_seq_len` and were truncated.
    pub truncated_count: usize,
    /// Percentage of truncated samples (0.0–100.0).
    pub truncated_pct: f64,
    /// Suggested `max_seq_len` — next power-of-two at or above the p95 original length,
    /// capped at the current `max_seq_len`.
    pub suggested_max_seq_len: usize,
    /// Column names found in the first record of the source file (populated by
    /// `peek_columns`; empty when constructed from an already-tokenized dataset).
    pub columns: Vec<String>,
}

/// Resolved dataset source — either a local path or HuggingFace dataset ID.
#[derive(Debug, Clone)]
pub enum DatasetSource {
    /// Local file or directory path.
    Local(PathBuf),
    /// HuggingFace dataset ID (e.g., "nohurry/Opus-4.6-Reasoning-3000x-filtered").
    HuggingFace(String),
}

/// Resolve a dataset source — either a local path or HuggingFace dataset ID.
///
/// HF IDs contain '/' but aren't existing file paths or relative paths.
pub fn resolve_dataset_source(source: &str) -> DatasetSource {
    let path = Path::new(source);
    if path.exists() {
        DatasetSource::Local(path.to_path_buf())
    } else if source.contains('/') && !source.starts_with('.') && !source.starts_with('/') {
        DatasetSource::HuggingFace(source.to_string())
    } else {
        DatasetSource::Local(path.to_path_buf()) // Let it error naturally on open
    }
}

/// `items` reordered so that position `p` holds what was at `order[p]`.
fn permute<T>(items: Vec<T>, order: &[usize]) -> Vec<T> {
    let mut slots: Vec<Option<T>> = items.into_iter().map(Some).collect();
    order
        .iter()
        .map(|&i| slots[i].take().expect("an order is a permutation"))
        .collect()
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Peek at the first row of a dataset (a file in any format
/// [`crate::rows`] reads, or a directory) and return its column names.
///
/// Useful for populating a column-picker UI or validating column config before
/// loading a large dataset. Returns an empty `Vec` if the dataset has no rows.
pub fn peek_columns(path: impl AsRef<Path>) -> Result<Vec<String>> {
    match RowReader::open(path)?.next() {
        Some(row) => Ok(row?.keys().cloned().collect()),
        None => Ok(Vec::new()),
    }
}

/// The number of characters of text in each of the first `limit` rows of a
/// dataset (all rows when `limit` is 0): every string a row holds, those
/// nested in a chat's messages included. A length estimate for sizing a
/// context before a dataset is tokenized (a token is roughly four characters
/// of English).
pub fn sample_row_chars(path: impl AsRef<Path>, limit: usize) -> Result<Vec<usize>> {
    fn chars(value: &serde_json::Value) -> usize {
        match value {
            serde_json::Value::String(s) => s.len(),
            serde_json::Value::Array(items) => items.iter().map(chars).sum(),
            serde_json::Value::Object(map) => map.values().map(chars).sum(),
            _ => 0,
        }
    }
    let rows = RowReader::open(path)?;
    let rows: Box<dyn Iterator<Item = _>> = if limit > 0 {
        Box::new(rows.take(limit))
    } else {
        Box::new(rows)
    };
    rows.map(|row| Ok(row?.values().map(chars).sum())).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_sample_creation() {
        let sample = Sample::new(vec![1, 2, 3, 4]);
        assert_eq!(sample.input_ids, vec![1, 2, 3, 4]);
        assert!(sample.labels.is_none());
    }

    #[test]
    fn test_sample_with_labels() {
        let sample = Sample::with_labels(vec![1, 2, 3], vec![-100, 2, 3]);
        assert_eq!(sample.input_ids, vec![1, 2, 3]);
        assert_eq!(sample.labels, Some(vec![-100, 2, 3]));
    }

    #[test]
    fn test_simple_jsonl_parsing() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, r#"{{"text": "Hello world"}}"#).unwrap();
        writeln!(file, r#"{{"text": "Another sample"}}"#).unwrap();

        let samples =
            TrainingDataset::load_jsonl_text(file.path(), DatasetFormat::Simple, None).unwrap();

        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].text, "Hello world");
        assert_eq!(samples[1].text, "Another sample");
    }

    #[test]
    fn test_alpaca_jsonl_parsing() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            r#"{{"instruction": "Say hello", "input": "", "output": "Hello!"}}"#
        )
        .unwrap();

        let samples =
            TrainingDataset::load_jsonl_text(file.path(), DatasetFormat::Alpaca, None).unwrap();

        assert_eq!(samples.len(), 1);
        assert!(samples[0].text.contains("Say hello"));
        assert!(samples[0].text.contains("Hello!"));
        assert_eq!(
            samples[0].prompt.as_deref(),
            Some("### Instruction:\nSay hello\n\n### Response:\n")
        );
    }

    #[test]
    fn test_auto_format_detection() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, r#"{{"text": "Auto detected"}}"#).unwrap();

        let samples =
            TrainingDataset::load_jsonl_text(file.path(), DatasetFormat::Auto, None).unwrap();

        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].text, "Auto detected");
    }

    /// Format detection reads the first row with content, not the first line.
    #[test]
    fn detection_skips_blank_lines() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file).unwrap();
        writeln!(file, r#"{{"instruction": "i", "output": "o"}}"#).unwrap();
        let samples =
            TrainingDataset::load_jsonl_text(file.path(), DatasetFormat::Auto, None).unwrap();
        assert_eq!(samples.len(), 1);
        assert!(samples[0].prompt.is_some());
    }

    #[test]
    fn test_train_val_split() {
        let samples: Vec<Sample> = (0..100).map(|i| Sample::new(vec![i as u32])).collect();
        let dataset = TrainingDataset::from_samples(samples);

        let (train, val) = dataset.train_val_split(0.2, 42);

        assert_eq!(train.len(), 80);
        assert_eq!(val.len(), 20);
    }

    /// A ratio outside `0..=1` used to underflow and panic.
    #[test]
    fn a_split_ratio_is_clamped() {
        let make = || {
            TrainingDataset::from_samples((0..10).map(|i| Sample::new(vec![i as u32])).collect())
        };
        let (train, val) = make().train_val_split(1.5, 1);
        assert_eq!((train.len(), val.len()), (0, 10));
        let (train, val) = make().train_val_split(-1.0, 1);
        assert_eq!((train.len(), val.len()), (10, 0));
    }

    /// Shuffling reorders the samples and their recorded lengths together,
    /// the same way for the same seed.
    #[test]
    fn shuffle_is_a_permutation_of_samples_and_lengths() {
        let make = || TrainingDataset {
            samples: (0..50).map(|i| Sample::new(vec![i as u32])).collect(),
            original_lengths: (0..50).collect(),
        };
        let (mut a, mut b) = (make(), make());
        a.shuffle(9);
        b.shuffle(9);
        let ids = |d: &TrainingDataset| -> Vec<u32> {
            d.samples.iter().map(|s| s.input_ids[0]).collect()
        };
        assert_eq!(ids(&a), ids(&b));
        assert_ne!(ids(&a), ids(&make()));
        let lengths: Vec<usize> = a.original_lengths.clone();
        assert_eq!(
            lengths,
            ids(&a).iter().map(|&i| i as usize).collect::<Vec<_>>()
        );
    }

    fn write_parquet(path: &Path, schema: Arc<arrow::datatypes::Schema>, batch: &RecordBatch) {
        use parquet::arrow::arrow_writer::ArrowWriter;
        let mut writer = ArrowWriter::try_new(File::create(path).unwrap(), schema, None).unwrap();
        writer.write(batch).unwrap();
        writer.close().unwrap();
    }

    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    #[test]
    fn test_parquet_loading() {
        use arrow::array::StringBuilder;
        use arrow::datatypes::{DataType, Field, Schema};

        let temp_file = tempfile::NamedTempFile::new().unwrap();
        let path = temp_file.path().to_path_buf();

        let schema = Arc::new(Schema::new(vec![
            Field::new("text", DataType::Utf8, false),
            Field::new("prompt", DataType::Utf8, true),
        ]));

        let mut text_builder = StringBuilder::new();
        text_builder.append_value("Hello world");
        text_builder.append_value("Another sample");
        text_builder.append_value("Third sample");
        let text_array = Arc::new(text_builder.finish());

        let mut prompt_builder = StringBuilder::new();
        prompt_builder.append_value("Hello ");
        prompt_builder.append_null();
        prompt_builder.append_value("Third ");
        let prompt_array = Arc::new(prompt_builder.finish());

        let batch = RecordBatch::try_new(schema.clone(), vec![text_array, prompt_array]).unwrap();
        write_parquet(&path, schema, &batch);

        // The file has no .parquet extension, which the reader does not need.
        let samples = TrainingDataset::load_parquet_text(&path, "text", None).unwrap();
        assert_eq!(samples.len(), 3);
        assert_eq!(samples[0].text, "Hello world");
        assert_eq!(samples[1].text, "Another sample");
        assert_eq!(samples[2].text, "Third sample");
        assert!(samples[0].prompt.is_none());

        let samples_with_prompt =
            TrainingDataset::load_parquet_text(&path, "text", Some("prompt")).unwrap();
        assert_eq!(samples_with_prompt.len(), 3);
        assert_eq!(samples_with_prompt[0].prompt, Some("Hello ".to_string()));
        assert!(samples_with_prompt[1].prompt.is_none()); // null in parquet
        assert_eq!(samples_with_prompt[2].prompt, Some("Third ".to_string()));
    }

    #[test]
    fn test_parquet_missing_column() {
        use arrow::array::StringBuilder;
        use arrow::datatypes::{DataType, Field, Schema};

        let temp_file = tempfile::NamedTempFile::new().unwrap();
        let path = temp_file.path().to_path_buf();
        let schema = Arc::new(Schema::new(vec![Field::new("body", DataType::Utf8, false)]));
        let mut builder = StringBuilder::new();
        builder.append_value("Test content");
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(builder.finish())]).unwrap();
        write_parquet(&path, schema, &batch);

        // Nothing in the file reads as a document, so detection fails and
        // names what the file has.
        let err = TrainingDataset::load_parquet_text(&path, "text", None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("body"), "{err}");
        // Asking for the column by name is the way in.
        let samples = TrainingDataset::load_parquet_text(&path, "body", None).unwrap();
        assert_eq!(samples[0].text, "Test content");
        let err = TrainingDataset::load_parquet_text(&path, "nope", Some("body"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("nope"), "{err}");
    }

    #[test]
    fn test_parquet_reasoning_format() {
        use arrow::array::StringBuilder;
        use arrow::datatypes::{DataType, Field, Schema};

        let temp_file = tempfile::NamedTempFile::new().unwrap();
        let path = temp_file.path().to_path_buf();

        let schema = Arc::new(Schema::new(vec![
            Field::new("problem", DataType::Utf8, false),
            Field::new("thinking", DataType::Utf8, true),
            Field::new("solution", DataType::Utf8, false),
        ]));

        let mut problem_builder = StringBuilder::new();
        problem_builder.append_value("What is 2+2?");
        problem_builder.append_value("Explain gravity.");
        let mut thinking_builder = StringBuilder::new();
        thinking_builder.append_value("Let me add 2 and 2.");
        thinking_builder.append_null(); // No thinking for second sample
        let mut solution_builder = StringBuilder::new();
        solution_builder.append_value("4");
        solution_builder.append_value("Gravity is a force of attraction.");

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(problem_builder.finish()),
                Arc::new(thinking_builder.finish()),
                Arc::new(solution_builder.finish()),
            ],
        )
        .unwrap();
        write_parquet(&path, schema, &batch);

        // Requesting "text" column should auto-detect reasoning format
        let samples = TrainingDataset::load_parquet_text(&path, "text", None).unwrap();
        assert_eq!(samples.len(), 2);

        assert_eq!(
            samples[0].text,
            "What is 2+2?\n\n<think>\nLet me add 2 and 2.\n</think>\n\n4"
        );
        // The problem and the blank line after it are what the model is
        // shown; the answer is what it writes.
        assert_eq!(samples[0].prompt, Some("What is 2+2?\n\n".to_string()));

        assert!(!samples[1].text.contains("<think>"));
        assert_eq!(
            samples[1].text,
            "Explain gravity.\n\nGravity is a force of attraction."
        );
        assert_eq!(samples[1].prompt, Some("Explain gravity.\n\n".to_string()));
    }

    #[test]
    fn peek_columns_reads_any_container() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, r#"{{"b": 1, "a": 2}}"#).unwrap();
        assert_eq!(peek_columns(file.path()).unwrap(), ["b", "a"]);
        let empty = NamedTempFile::new().unwrap();
        assert!(peek_columns(empty.path()).unwrap().is_empty());
    }

    /// One row, read and rendered on its own: the layout is the row's, the
    /// text the chat template's, and a row of no known layout is an error
    /// that names its columns.
    #[test]
    fn a_row_renders_on_its_own() {
        let row: Row = serde_json::from_str(
            r#"{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}"#,
        )
        .unwrap();
        let template = ChatTemplate::chatml();
        let samples = DatasetFormat::Auto
            .render_row(&row, Some(&template))
            .unwrap();
        assert_eq!(samples.len(), 1);
        assert!(samples[0].templated);
        assert!(samples[0].text.contains("<|im_start|>user\nhi<|im_end|>"));
        assert!(samples[0].text.contains("<|im_start|>assistant\nyo"));
        // Without a template, the plain blocks.
        let plain = DatasetFormat::Auto.render_row(&row, None).unwrap();
        assert_eq!(plain[0].text, "User: hi\n\nAssistant: yo\n\n");

        let odd: Row = serde_json::from_str(r#"{"weird": 1}"#).unwrap();
        let err = DatasetFormat::Auto
            .render_row(&odd, None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("weird"), "{err}");
    }

    /// The length a sample has in training: the tokenizer's special tokens
    /// for text no template rendered, an end-of-sequence token after it.
    #[test]
    #[allow(clippy::single_range_in_vec_init)] // one trained span
    fn a_sample_counts_the_tokens_training_would_see() {
        let tokenizer = crate::tokenizer::testing::word_level(
            &["<unk>", "a", "b"],
            &[("<s>", true), ("</s>", true)],
        );
        let document = TextSample::document("a b a");
        let rendered = TextSample::with_spans("a b a".into(), vec![2..5], true);
        // Rendered text is counted as it stands.
        assert_eq!(rendered.token_len(&tokenizer).unwrap(), 3);
        // A document carries nothing the rendered text does not, so it is
        // never shorter.
        assert!(document.token_len(&tokenizer).unwrap() >= 3);
    }

    #[test]
    fn an_instruction_row_is_put_in_the_chat_format_on_request() {
        let row: Row =
            serde_json::from_str(r#"{"instruction": "Say hi", "output": "Hello!"}"#).unwrap();
        let template = ChatTemplate::chatml();
        // As training reads it: the Alpaca layout.
        let as_trained = DatasetFormat::Auto
            .render_row(&row, Some(&template))
            .unwrap();
        assert!(as_trained[0].text.starts_with("### Instruction:"));
        // As a chat: the model's format, the answer the span the loss covers.
        let as_chat = DatasetFormat::Auto
            .render_row_as_chat(&row, Some(&template))
            .unwrap();
        assert!(
            as_chat[0]
                .text
                .contains("<|im_start|>user\nSay hi<|im_end|>")
        );
        assert!(as_chat[0].templated);
        let span = as_chat[0].trained.as_ref().unwrap()[0].clone();
        assert!(as_chat[0].text[span].starts_with("Hello!"));
    }
}
