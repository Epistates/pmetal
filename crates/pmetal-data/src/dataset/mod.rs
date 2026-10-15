//! Dataset types and loading.

use super::chat_templates::{ChatTemplate, Message};
use arrow::array::{Array as ArrowArray, StringArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use pmetal_core::Result;
use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

/// A training sample.
#[derive(Debug, Clone)]
pub struct Sample {
    /// Input token IDs.
    pub input_ids: Vec<u32>,
    /// Attention mask.
    pub attention_mask: Vec<u32>,
    /// Labels (for supervised training). Use -100 for tokens to ignore in loss.
    pub labels: Option<Vec<i64>>,
    /// Image paths (for multimodal training).
    pub images: Option<Vec<PathBuf>>,
}

impl Sample {
    /// Create a new sample with input_ids only.
    pub fn new(input_ids: Vec<u32>) -> Self {
        let len = input_ids.len();
        Self {
            input_ids,
            attention_mask: vec![1; len],
            labels: None,
            images: None,
        }
    }

    /// Create a sample with input_ids and labels.
    pub fn with_labels(input_ids: Vec<u32>, labels: Vec<i64>) -> Self {
        let len = input_ids.len();
        Self {
            input_ids,
            attention_mask: vec![1; len],
            labels: Some(labels),
            images: None,
        }
    }

    /// Create a sample with input_ids, labels, and images.
    pub fn with_images(input_ids: Vec<u32>, labels: Vec<i64>, images: Vec<PathBuf>) -> Self {
        let len = input_ids.len();
        Self {
            input_ids,
            attention_mask: vec![1; len],
            labels: Some(labels),
            images: Some(images),
        }
    }
}

/// Raw text sample from JSONL files.
#[derive(Debug, Clone)]
pub struct TextSample {
    /// The full text to tokenize.
    pub text: String,
    /// Optional prompt portion (for SFT, the part before the response).
    pub prompt: Option<String>,
}

/// Alpaca-style JSONL format.
#[derive(Debug, Deserialize)]
struct AlpacaFormat {
    instruction: String,
    #[serde(default)]
    input: String,
    output: String,
}

/// Simple text JSONL format.
#[derive(Debug, Deserialize)]
struct SimpleFormat {
    text: String,
}

/// OpenAI-style messages format.
#[derive(Debug, Deserialize)]
struct MessagesFormat {
    messages: Vec<OpenAiMessage>,
}

/// OpenAI-style message.
#[derive(Debug, Deserialize)]
struct OpenAiMessage {
    role: String,
    content: String,
}

/// ShareGPT conversation message.
#[derive(Debug, Deserialize)]
struct ShareGptMessage {
    from: String,
    value: String,
}

/// ShareGPT-style JSONL format.
#[derive(Debug, Deserialize)]
struct ShareGptFormat {
    conversations: Vec<ShareGptMessage>,
}

/// Dataset format variants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DatasetFormat {
    /// Simple format: {"text": "..."}
    Simple,
    /// Alpaca format: {"instruction": "...", "input": "...", "output": "..."}
    Alpaca,
    /// ShareGPT format: {"conversations": [{"from": "human", "value": "..."}, ...]}
    ShareGpt,
    /// OpenAI format: {"messages": [{"role": "user", "content": "..."}, ...]}
    OpenAi,
    /// Reasoning format: {"problem": "...", "thinking": "...", "solution": "..."}
    Reasoning,
    /// Auto-detect format from first line
    Auto,
    /// Custom column extraction — use arbitrary JSON field names.
    Custom {
        /// Field name for the text content (or combined prompt+response).
        text_column: String,
        /// Multiple columns to concatenate (e.g. `["thinking", "solution"]`).
        /// Takes precedence over `text_column` when non-empty.
        text_columns: Option<Vec<String>>,
        /// Separator between concatenated columns (default: `"\n\n"`).
        column_separator: String,
        /// Optional: field for the prompt portion (masked from loss).
        prompt_column: Option<String>,
        /// Optional: separate response column.
        response_column: Option<String>,
    },
}

/// Configuration for custom column extraction from JSONL/Parquet datasets.
///
/// Use this to specify which JSON fields contain the training text, and optionally
/// which field contains the prompt (for loss masking) vs the response.
#[derive(Debug, Clone, Default)]
pub struct DatasetColumnConfig {
    /// Main text/content column name. When set and differs from "text", overrides
    /// auto-detection and uses custom column extraction.
    pub text_column: Option<String>,
    /// Multiple text columns to concatenate (e.g. `["thinking", "solution"]`).
    /// Joined with `column_separator` between each. Takes precedence over `text_column`.
    pub text_columns: Option<Vec<String>>,
    /// Separator between concatenated columns (default: `"\n\n"`).
    pub column_separator: Option<String>,
    /// Prompt column (masked from loss). When set, only response tokens contribute
    /// to the training loss.
    pub prompt_column: Option<String>,
    /// Response column. When set together with `prompt_column`, the full sequence
    /// is `prompt || response` and the prompt portion receives label -100.
    pub response_column: Option<String>,
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

/// Dataset for training.
#[derive(Clone)]
pub struct TrainingDataset {
    samples: Vec<Sample>,
    /// Original token lengths before truncation, recorded during `from_jsonl_tokenized`.
    /// Used by `compute_statistics` for accurate truncation reporting.
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

    /// Load and tokenize a dataset from a JSONL file.
    ///
    /// # Arguments
    /// * `path` - Path to the JSONL file
    /// * `tokenizer` - Tokenizer to use
    /// * `format` - Dataset format (or Auto to detect)
    /// * `max_length` - Maximum sequence length
    /// * `template` - Optional chat template for formatting (OpenAI/ShareGPT formats)
    /// * `columns` - Optional custom column config; when provided with a `text_column`,
    ///   overrides format detection and uses custom column extraction.
    pub fn from_jsonl_tokenized<P: AsRef<Path>>(
        path: P,
        tokenizer: &super::Tokenizer,
        format: DatasetFormat,
        max_length: usize,
        template: Option<&ChatTemplate>,
        columns: Option<&DatasetColumnConfig>,
    ) -> Result<Self> {
        // If custom columns are specified, build the effective format.
        // Special case: if columns match the reasoning schema (thinking/solution +
        // problem prompt), route to Reasoning format to get proper <think> tags and
        // correct loss masking.
        let effective_format = if let Some(col_cfg) = columns {
            let has_multi = col_cfg.text_columns.as_ref().is_some_and(|v| !v.is_empty());
            let has_custom_single = col_cfg.text_column.as_ref().is_some_and(|t| t != "text");

            // Detect reasoning-pattern columns: text includes "thinking" or "solution",
            // prompt is "problem"
            let is_reasoning_pattern = has_multi
                && col_cfg.prompt_column.as_deref() == Some("problem")
                && col_cfg
                    .text_columns
                    .as_ref()
                    .is_some_and(|cols| cols.iter().any(|c| c == "thinking" || c == "solution"));

            if is_reasoning_pattern {
                tracing::info!(
                    "Detected reasoning-pattern columns (thinking/solution + problem); \
                     using Reasoning format for proper <think> tags and loss masking"
                );
                DatasetFormat::Reasoning
            } else if has_multi || has_custom_single {
                DatasetFormat::Custom {
                    text_column: col_cfg
                        .text_column
                        .clone()
                        .unwrap_or_else(|| "text".to_string()),
                    text_columns: col_cfg.text_columns.clone(),
                    column_separator: col_cfg
                        .column_separator
                        .clone()
                        .unwrap_or_else(|| "\n\n".to_string()),
                    prompt_column: col_cfg.prompt_column.clone(),
                    response_column: col_cfg.response_column.clone(),
                }
            } else {
                format
            }
        } else {
            format
        };

        let text_samples = Self::load_jsonl_text(path, effective_format, template)?;
        let n_samples = text_samples.len();

        // Parallel tokenization with rayon — tokenizers::Tokenizer is Send+Sync
        use rayon::prelude::*;

        let results: Vec<Result<(Sample, usize)>> = text_samples
            .into_par_iter()
            .map(|text_sample| {
                // Encode once, use for both orig_len tracking and the sample
                let mut input_ids = tokenizer.encode_with_special_tokens(&text_sample.text)?;
                let orig_len = input_ids.len();

                if input_ids.len() > max_length {
                    input_ids.truncate(max_length);
                }

                let labels = if let Some(ref prompt) = text_sample.prompt {
                    let prompt_ids = tokenizer.encode_with_special_tokens(prompt)?;
                    let prompt_len = prompt_ids.len().min(input_ids.len());
                    let mut labels: Vec<i64> = input_ids.iter().map(|&id| id as i64).collect();
                    for label in labels.iter_mut().take(prompt_len) {
                        *label = -100;
                    }
                    Some(labels)
                } else {
                    Some(input_ids.iter().map(|&id| id as i64).collect())
                };

                let sample = Sample {
                    attention_mask: vec![1; input_ids.len()],
                    input_ids,
                    labels,
                    images: None,
                };
                Ok((sample, orig_len))
            })
            .collect();

        let mut samples = Vec::with_capacity(n_samples);
        let mut original_lengths = Vec::with_capacity(n_samples);
        for result in results {
            let (sample, orig_len) = result?;
            samples.push(sample);
            original_lengths.push(orig_len);
        }

        Ok(Self {
            samples,
            original_lengths,
        })
    }

    /// Load raw text samples from JSONL without tokenizing.
    ///
    /// If `path` is a directory, auto-discovers a `.jsonl` file inside it:
    /// tries `train.jsonl`, `data.jsonl`, `dataset.jsonl` in order, then
    /// suggests any other `.jsonl` files found.
    pub fn load_jsonl_text<P: AsRef<Path>>(
        path: P,
        format: DatasetFormat,
        template: Option<&ChatTemplate>,
    ) -> Result<Vec<TextSample>> {
        let path = Self::resolve_dataset_path(path.as_ref())?;

        let file = File::open(&path).map_err(|e| {
            pmetal_core::PMetalError::Io(std::io::Error::new(
                e.kind(),
                format!("Failed to open dataset file '{}': {}", path.display(), e),
            ))
        })?;

        let reader = BufReader::new(file);
        let mut samples = Vec::new();
        let mut detected_format = format;
        let mut first_content_seen = false;

        for (line_num, line_result) in reader.lines().enumerate() {
            let line = line_result.map_err(|e| {
                pmetal_core::PMetalError::Io(std::io::Error::new(
                    e.kind(),
                    format!("Failed to read line {}: {}", line_num + 1, e),
                ))
            })?;

            if line.trim().is_empty() {
                continue;
            }

            // Auto-detect format on the first non-empty line (not necessarily line 0,
            // which may be blank and cause a false-negative format detection).
            if !first_content_seen && detected_format == DatasetFormat::Auto {
                first_content_seen = true;
                detected_format = Self::detect_format(&line)?;
            } else {
                first_content_seen = true;
            }

            let sample = Self::parse_line(&line, &detected_format, line_num, template)?;
            samples.push(sample);
        }

        Ok(samples)
    }

    /// Resolve a dataset path, handling the case where the user passes a directory.
    fn resolve_dataset_path(path: &Path) -> Result<PathBuf> {
        if !path.is_dir() {
            return Ok(path.to_path_buf());
        }

        // HF cache structure: datasets--org--name/snapshots/{hash}/ — traverse into it
        let search_dirs = Self::collect_search_dirs(path);

        // Data file extensions in priority order
        const EXTS: &[&str] = &[".jsonl", ".json", ".parquet", ".csv", ".arrow"];

        // Try well-known stems first across all search dirs
        const WELL_KNOWN_STEMS: &[&str] = &["train", "data", "dataset"];
        for dir in &search_dirs {
            for stem in WELL_KNOWN_STEMS {
                for ext in EXTS {
                    let candidate = dir.join(format!("{stem}{ext}"));
                    if candidate.exists() {
                        tracing::info!("Auto-discovered dataset file: {}", candidate.display());
                        return Ok(candidate);
                    }
                }
            }
        }

        // Fallback: find any data file across all search dirs
        for dir in &search_dirs {
            if let Ok(mut found) = Self::find_data_files(dir) {
                // Prefer jsonl > json > parquet > csv
                found.sort_by_key(|p| {
                    let s = p.to_string_lossy();
                    if s.ends_with(".jsonl") {
                        0
                    } else if s.ends_with(".json") {
                        1
                    } else if s.ends_with(".parquet") {
                        2
                    } else if s.ends_with(".csv") {
                        3
                    } else {
                        4
                    }
                });
                if let Some(best) = found.into_iter().next() {
                    tracing::info!("Auto-discovered dataset file: {}", best.display());
                    return Ok(best);
                }
            }
        }

        Err(pmetal_core::PMetalError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!(
                "'{}' is a directory with no recognized data files (.jsonl, .json, .parquet, .csv). \
                 Pass a file path directly instead.",
                path.display(),
            ),
        )))
    }

    /// Collect directories to search for data files.
    /// Handles HF cache layout: `snapshots/{hash}/` and `data/`.
    fn collect_search_dirs(root: &Path) -> Vec<PathBuf> {
        let mut dirs = vec![root.to_path_buf()];

        // HF cache: snapshots/{hash}/ — use the most recent snapshot
        let snapshots = root.join("snapshots");
        if snapshots.is_dir() {
            if let Ok(rd) = std::fs::read_dir(&snapshots) {
                let mut snap_dirs: Vec<PathBuf> = rd
                    .filter_map(|e| e.ok())
                    .filter(|e| e.path().is_dir())
                    .map(|e| e.path())
                    .collect();
                // Sort by modification time descending (most recent first)
                snap_dirs.sort_by(|a, b| {
                    let t_a = a
                        .metadata()
                        .and_then(|m| m.modified())
                        .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
                    let t_b = b
                        .metadata()
                        .and_then(|m| m.modified())
                        .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
                    t_b.cmp(&t_a)
                });
                dirs.extend(snap_dirs);
            }
        }

        // Common subdirectories
        for sub in &["data", "train"] {
            let d = root.join(sub);
            if d.is_dir() {
                dirs.push(d);
            }
        }

        dirs
    }

    /// Find all data files in a directory (non-recursive, single level).
    fn find_data_files(dir: &Path) -> Result<Vec<PathBuf>> {
        let rd = std::fs::read_dir(dir).map_err(|e| {
            pmetal_core::PMetalError::Io(std::io::Error::new(
                e.kind(),
                format!("Failed to read directory '{}': {}", dir.display(), e),
            ))
        })?;
        Ok(rd
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let p = entry.path();
                if p.is_file() {
                    let name = p.file_name()?.to_string_lossy().to_lowercase();
                    if name.ends_with(".jsonl")
                        || name.ends_with(".json")
                        || name.ends_with(".parquet")
                        || name.ends_with(".csv")
                        || name.ends_with(".arrow")
                    {
                        return Some(p);
                    }
                }
                // Follow symlinks (HF cache uses them)
                if entry.file_type().ok()?.is_symlink() {
                    let real = std::fs::metadata(&p).ok()?;
                    if real.is_file() {
                        let name = p.file_name()?.to_string_lossy().to_lowercase();
                        if name.ends_with(".jsonl")
                            || name.ends_with(".json")
                            || name.ends_with(".parquet")
                            || name.ends_with(".csv")
                            || name.ends_with(".arrow")
                        {
                            return Some(p);
                        }
                    }
                }
                None
            })
            .collect())
    }

    /// Public wrapper for resolve_dataset_path, used by CLI for HF dataset resolution.
    pub fn resolve_dataset_path_pub(path: &Path) -> Result<PathBuf> {
        Self::resolve_dataset_path(path)
    }

    /// Detect the format from a JSON line.
    fn detect_format(line: &str) -> Result<DatasetFormat> {
        let value: serde_json::Value = serde_json::from_str(line).map_err(|e| {
            pmetal_core::PMetalError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Invalid JSON: {}", e),
            ))
        })?;

        if value.get("text").is_some() {
            Ok(DatasetFormat::Simple)
        } else if value.get("instruction").is_some() {
            Ok(DatasetFormat::Alpaca)
        } else if value.get("conversations").is_some() {
            Ok(DatasetFormat::ShareGpt)
        } else if value.get("messages").is_some() {
            Ok(DatasetFormat::OpenAi)
        } else if value.get("problem").is_some() || value.get("thinking").is_some() {
            Ok(DatasetFormat::Reasoning)
        } else {
            Err(pmetal_core::PMetalError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Could not detect dataset format. Expected 'text', 'instruction', 'conversations', 'messages', or 'problem' field.",
            )))
        }
    }

    /// Extract text content from a JSON value, handling different types:
    /// - String → return as-is
    /// - Array of message objects (OpenAI/ShareGPT chat format) → concatenate content fields
    /// - Other → serialize to JSON string
    fn extract_text_from_value(val: &serde_json::Value) -> String {
        match val {
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Array(arr) => {
                // Chat format: [{"role": "user", "content": "..."}, ...]
                let mut parts = Vec::new();
                for item in arr {
                    if let Some(content) = item.get("content").or(item.get("value")) {
                        if let Some(s) = content.as_str() {
                            // Optionally prefix with role
                            if let Some(role) = item
                                .get("role")
                                .or(item.get("from"))
                                .and_then(|r| r.as_str())
                            {
                                parts.push(format!("{}: {}", role, s));
                            } else {
                                parts.push(s.to_string());
                            }
                        }
                    }
                }
                if parts.is_empty() {
                    // Fallback: serialize array to string
                    serde_json::to_string(val).unwrap_or_default()
                } else {
                    parts.join("\n")
                }
            }
            serde_json::Value::Number(n) => n.to_string(),
            serde_json::Value::Bool(b) => b.to_string(),
            _ => serde_json::to_string(val).unwrap_or_default(),
        }
    }

    /// Parse a single JSONL line using custom column names.
    ///
    /// Returns `(full_text, Option<prompt_text>)` where `prompt_text` is the
    /// portion that should be loss-masked (label = -100).
    fn parse_custom_line(
        line: &str,
        text_col: &str,
        text_cols: Option<&[String]>,
        separator: &str,
        prompt_col: Option<&str>,
        response_col: Option<&str>,
    ) -> Result<(String, Option<String>)> {
        let obj: serde_json::Value = serde_json::from_str(line).map_err(|e| {
            pmetal_core::PMetalError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Invalid JSON: {}", e),
            ))
        })?;

        // Multi-column concatenation takes precedence
        if let Some(cols) = text_cols {
            if !cols.is_empty() {
                let parts: Vec<String> = cols
                    .iter()
                    .filter_map(|c| {
                        let val = obj.get(c.as_str())?;
                        Some(Self::extract_text_from_value(val))
                    })
                    .collect();
                if parts.is_empty() {
                    return Err(pmetal_core::PMetalError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!(
                            "None of the requested columns {:?} were present in this JSON line; \
                             check that at least one column name matches the dataset schema",
                            cols
                        ),
                    )));
                }
                let text = parts.join(separator);
                // If prompt_col is set, mask that portion
                if let Some(pc) = prompt_col {
                    let prompt = obj.get(pc).and_then(|v| v.as_str()).map(|s| s.to_string());
                    if let Some(ref prompt_text) = prompt {
                        // If the prompt column is NOT one of the text columns, we must
                        // prepend it to the text so that loss masking aligns correctly.
                        // Otherwise the first N tokens of the text get masked even though
                        // they don't correspond to the prompt content.
                        let prompt_is_in_text_cols = text_cols
                            .map(|cols| cols.iter().any(|c| c.as_str() == pc))
                            .unwrap_or(false);
                        if !prompt_is_in_text_cols {
                            let full_text = format!("{}{}{}", prompt_text, separator, text);
                            return Ok((full_text, prompt));
                        }
                    }
                    return Ok((text, prompt));
                }
                return Ok((text, None));
            }
        }

        if let (Some(pc), Some(rc)) = (prompt_col, response_col) {
            let prompt = obj
                .get(pc)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let response = obj
                .get(rc)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            Ok((format!("{}{}", prompt, response), Some(prompt)))
        } else {
            let text = obj
                .get(text_col)
                .map(Self::extract_text_from_value)
                .ok_or_else(|| {
                    pmetal_core::PMetalError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Column '{}' not found in JSON line", text_col),
                    ))
                })?;
            if let Some(pc) = prompt_col {
                let prompt = obj.get(pc).and_then(|v| v.as_str()).map(|s| s.to_string());
                Ok((text, prompt))
            } else {
                Ok((text, None))
            }
        }
    }

    /// Parse a single JSONL line.
    fn parse_line(
        line: &str,
        format: &DatasetFormat,
        line_num: usize,
        template: Option<&ChatTemplate>,
    ) -> Result<TextSample> {
        match format {
            DatasetFormat::Simple => {
                let parsed: SimpleFormat = serde_json::from_str(line).map_err(|e| {
                    pmetal_core::PMetalError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Line {}: {}", line_num + 1, e),
                    ))
                })?;
                Ok(TextSample {
                    text: parsed.text,
                    prompt: None,
                })
            }
            DatasetFormat::Alpaca => {
                let parsed: AlpacaFormat = serde_json::from_str(line).map_err(|e| {
                    pmetal_core::PMetalError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Line {}: {}", line_num + 1, e),
                    ))
                })?;

                // Format: instruction + optional input -> output
                let prompt = if parsed.input.is_empty() {
                    format!(
                        "### Instruction:\n{}\n\n### Response:\n",
                        parsed.instruction
                    )
                } else {
                    format!(
                        "### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n",
                        parsed.instruction, parsed.input
                    )
                };

                let full_text = format!("{}{}", prompt, parsed.output);
                Ok(TextSample {
                    text: full_text,
                    prompt: Some(prompt),
                })
            }
            DatasetFormat::ShareGpt => {
                let parsed: ShareGptFormat = serde_json::from_str(line).map_err(|e| {
                    pmetal_core::PMetalError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Line {}: {}", line_num + 1, e),
                    ))
                })?;

                if let Some(tmpl) = template {
                    let messages: Vec<Message> = parsed
                        .conversations
                        .iter()
                        .map(|m| {
                            let role = match m.from.as_str() {
                                "human" | "user" | "system" => m.from.as_str(),
                                "gpt" | "assistant" => "assistant",
                                other => other,
                            };
                            Message::new(role, &m.value)
                        })
                        .collect();

                    let formatted = tmpl.apply(&messages);
                    let prompt = formatted.prompt().to_string();
                    Ok(TextSample {
                        text: formatted.text,
                        prompt: Some(prompt),
                    })
                } else {
                    // Fallback to basic formatting if no template
                    let mut full_text = String::new();
                    let mut prompt_end = 0;

                    for msg in parsed.conversations.iter() {
                        let role = match msg.from.as_str() {
                            "human" | "user" => "User",
                            "gpt" | "assistant" => "Assistant",
                            "system" => "System",
                            other => other,
                        };

                        if msg.from == "human" || msg.from == "user" || msg.from == "system" {
                            full_text.push_str(&format!("{}: {}\n\n", role, msg.value));
                            prompt_end = full_text.len();
                        } else {
                            full_text.push_str(&format!("{}: {}\n\n", role, msg.value));
                        }
                    }

                    let prompt = if prompt_end > 0 {
                        Some(full_text[..prompt_end].to_string())
                    } else {
                        None
                    };

                    Ok(TextSample {
                        text: full_text,
                        prompt,
                    })
                }
            }
            DatasetFormat::OpenAi => {
                let parsed: MessagesFormat = serde_json::from_str(line).map_err(|e| {
                    pmetal_core::PMetalError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Line {}: {}", line_num + 1, e),
                    ))
                })?;

                if let Some(tmpl) = template {
                    let messages: Vec<Message> = parsed
                        .messages
                        .iter()
                        .map(|m| Message::new(&m.role, &m.content))
                        .collect();

                    let formatted = tmpl.apply(&messages);
                    let prompt = formatted.prompt().to_string();
                    Ok(TextSample {
                        text: formatted.text,
                        prompt: Some(prompt),
                    })
                } else {
                    let mut full_text = String::new();
                    let mut prompt_end = 0;

                    for msg in parsed.messages.iter() {
                        let role = match msg.role.as_str() {
                            "user" => "User",
                            "assistant" => "Assistant",
                            "system" => "System",
                            other => other,
                        };

                        if msg.role == "user" || msg.role == "system" {
                            full_text.push_str(&format!("{}: {}\n\n", role, msg.content));
                            prompt_end = full_text.len();
                        } else {
                            full_text.push_str(&format!("{}: {}\n\n", role, msg.content));
                        }
                    }

                    let prompt = if prompt_end > 0 {
                        Some(full_text[..prompt_end].to_string())
                    } else {
                        None
                    };

                    Ok(TextSample {
                        text: full_text,
                        prompt,
                    })
                }
            }
            DatasetFormat::Reasoning => {
                let obj: serde_json::Value = serde_json::from_str(line).map_err(|e| {
                    pmetal_core::PMetalError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Line {}: {}", line_num + 1, e),
                    ))
                })?;

                let problem = obj.get("problem").and_then(|v| v.as_str()).unwrap_or("");
                let thinking = obj.get("thinking").and_then(|v| v.as_str()).unwrap_or("");
                let solution = obj.get("solution").and_then(|v| v.as_str()).unwrap_or("");

                let prompt = problem.to_string();
                let response = if thinking.is_empty() {
                    solution.to_string()
                } else {
                    format!("<think>\n{}\n</think>\n\n{}", thinking, solution)
                };

                Ok(TextSample {
                    text: format!("{}\n\n{}", prompt, response),
                    prompt: Some(prompt),
                })
            }
            DatasetFormat::Auto => {
                // Should not reach here after detection
                unreachable!("Auto format should be resolved before parsing")
            }
            DatasetFormat::Custom {
                text_column,
                text_columns,
                column_separator,
                prompt_column,
                response_column,
            } => {
                let (text, prompt) = Self::parse_custom_line(
                    line,
                    text_column,
                    text_columns.as_deref(),
                    column_separator,
                    prompt_column.as_deref(),
                    response_column.as_deref(),
                )
                .map_err(|e| {
                    // Annotate with line number for better diagnostics.
                    pmetal_core::PMetalError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Line {}: {}", line_num + 1, e),
                    ))
                })?;
                Ok(TextSample { text, prompt })
            }
        }
    }

    /// Tokenize a text sample into a training sample.
    fn tokenize_sample(
        text_sample: &TextSample,
        tokenizer: &super::Tokenizer,
        max_length: usize,
    ) -> Result<Sample> {
        let mut input_ids = tokenizer.encode_with_special_tokens(&text_sample.text)?;

        // Truncate if needed
        if input_ids.len() > max_length {
            input_ids.truncate(max_length);
        }

        // Create labels: mask prompt tokens with -100, keep response tokens
        let labels = if let Some(ref prompt) = text_sample.prompt {
            let prompt_ids = tokenizer.encode_with_special_tokens(prompt)?;
            let prompt_len = prompt_ids.len().min(input_ids.len());

            let mut labels: Vec<i64> = input_ids.iter().map(|&id| id as i64).collect();
            // Mask prompt tokens with -100 (ignored in loss)
            for label in labels.iter_mut().take(prompt_len) {
                *label = -100;
            }
            Some(labels)
        } else {
            // For simple format, all tokens are labels (shifted by model)
            Some(input_ids.iter().map(|&id| id as i64).collect())
        };

        Ok(Sample {
            attention_mask: vec![1; input_ids.len()],
            input_ids,
            labels,
            images: None,
        })
    }

    /// Load dataset from a JSONL file (legacy, loads as empty).
    #[deprecated(note = "Use from_jsonl_tokenized instead")]
    pub fn from_jsonl<P: AsRef<Path>>(_path: P) -> Result<Self> {
        Ok(Self::new())
    }

    /// Load and tokenize a dataset from a Parquet file.
    ///
    /// # Arguments
    /// * `path` - Path to the Parquet file
    /// * `tokenizer` - Tokenizer to use
    /// * `text_column` - Name of the column containing text (e.g., "text", "content")
    /// * `max_length` - Maximum sequence length
    /// * `prompt_column` - Optional column name for prompt portion (for SFT label masking)
    pub fn from_parquet_tokenized<P: AsRef<Path>>(
        path: P,
        tokenizer: &super::Tokenizer,
        text_column: &str,
        max_length: usize,
        prompt_column: Option<&str>,
    ) -> Result<Self> {
        let text_samples = Self::load_parquet_text(path, text_column, prompt_column)?;
        let mut samples = Vec::with_capacity(text_samples.len());

        for text_sample in text_samples {
            let sample = Self::tokenize_sample(&text_sample, tokenizer, max_length)?;
            samples.push(sample);
        }

        Ok(Self {
            samples,
            original_lengths: Vec::new(),
        })
    }

    /// Load raw text samples from a Parquet file without tokenizing.
    ///
    /// # Arguments
    /// * `path` - Path to the Parquet file
    /// * `text_column` - Name of the column containing text
    /// * `prompt_column` - Optional column name for prompt portion
    pub fn load_parquet_text<P: AsRef<Path>>(
        path: P,
        text_column: &str,
        prompt_column: Option<&str>,
    ) -> Result<Vec<TextSample>> {
        let file = File::open(path.as_ref()).map_err(|e| {
            pmetal_core::PMetalError::Io(std::io::Error::new(
                e.kind(),
                format!("Failed to open Parquet file: {}", e),
            ))
        })?;

        let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| {
            pmetal_core::PMetalError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Failed to create Parquet reader: {}", e),
            ))
        })?;

        let schema = builder.schema();

        // Find column indices — if primary text column not found, try reasoning format
        let text_idx_result = schema.index_of(text_column);

        if text_idx_result.is_err() {
            // Check for reasoning format columns (problem/solution)
            let has_problem = schema.index_of("problem").is_ok();
            let has_solution = schema.index_of("solution").is_ok();

            if has_problem && has_solution {
                tracing::info!(
                    "Column '{}' not found, auto-detected reasoning format (problem/thinking/solution)",
                    text_column
                );
                // Rebuild the reader (builder was consumed by schema())
                let file2 = File::open(path.as_ref()).map_err(|e| {
                    pmetal_core::PMetalError::Io(std::io::Error::new(
                        e.kind(),
                        format!("Failed to reopen Parquet file: {}", e),
                    ))
                })?;
                return Self::load_parquet_reasoning(file2);
            }
        }

        let text_idx = text_idx_result.map_err(|_| {
            pmetal_core::PMetalError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Column '{}' not found in Parquet file. Available columns: {:?}",
                    text_column,
                    schema.fields().iter().map(|f| f.name()).collect::<Vec<_>>()
                ),
            ))
        })?;

        let prompt_idx = if let Some(prompt_col) = prompt_column {
            Some(schema.index_of(prompt_col).map_err(|_| {
                pmetal_core::PMetalError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Prompt column '{}' not found in Parquet file", prompt_col),
                ))
            })?)
        } else {
            None
        };

        let reader = builder.build().map_err(|e| {
            pmetal_core::PMetalError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Failed to build Parquet reader: {}", e),
            ))
        })?;

        let mut samples = Vec::new();

        for batch_result in reader {
            let batch = batch_result.map_err(|e| {
                pmetal_core::PMetalError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Failed to read Parquet batch: {}", e),
                ))
            })?;

            let text_col = batch.column(text_idx);
            let text_array = text_col
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| {
                    pmetal_core::PMetalError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Column '{}' is not a string type", text_column),
                    ))
                })?;

            let prompt_array = if let Some(idx) = prompt_idx {
                let col = batch.column(idx);
                Some(col.as_any().downcast_ref::<StringArray>().ok_or_else(|| {
                    pmetal_core::PMetalError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "Prompt column is not a string type",
                    ))
                })?)
            } else {
                None
            };

            for i in 0..batch.num_rows() {
                if text_array.is_null(i) {
                    continue;
                }

                let text = text_array.value(i).to_string();
                let prompt = prompt_array.and_then(|arr| {
                    if arr.is_null(i) {
                        None
                    } else {
                        Some(arr.value(i).to_string())
                    }
                });

                samples.push(TextSample { text, prompt });
            }
        }

        Ok(samples)
    }

    /// Load reasoning format samples from a Parquet file.
    ///
    /// Reads `problem`, `thinking` (optional), and `solution` columns,
    /// composing text the same way as JSONL `DatasetFormat::Reasoning`:
    /// `text = "{problem}\n\n<think>\n{thinking}\n</think>\n\n{solution}"`
    fn load_parquet_reasoning(file: File) -> Result<Vec<TextSample>> {
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| {
            pmetal_core::PMetalError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Failed to create Parquet reader: {}", e),
            ))
        })?;

        let schema = builder.schema();
        let problem_idx = schema.index_of("problem").map_err(|_| {
            pmetal_core::PMetalError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Missing 'problem' column in reasoning Parquet",
            ))
        })?;
        let solution_idx = schema.index_of("solution").map_err(|_| {
            pmetal_core::PMetalError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Missing 'solution' column in reasoning Parquet",
            ))
        })?;
        let thinking_idx = schema.index_of("thinking").ok();

        let reader = builder.build().map_err(|e| {
            pmetal_core::PMetalError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Failed to build Parquet reader: {}", e),
            ))
        })?;

        let mut samples = Vec::new();

        for batch_result in reader {
            let batch = batch_result.map_err(|e| {
                pmetal_core::PMetalError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Failed to read Parquet batch: {}", e),
                ))
            })?;

            let problem_arr = batch
                .column(problem_idx)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| {
                    pmetal_core::PMetalError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "'problem' column is not a string type",
                    ))
                })?;

            let solution_arr = batch
                .column(solution_idx)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| {
                    pmetal_core::PMetalError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "'solution' column is not a string type",
                    ))
                })?;

            let thinking_arr =
                thinking_idx.map(|idx| batch.column(idx).as_any().downcast_ref::<StringArray>());

            for i in 0..batch.num_rows() {
                if problem_arr.is_null(i) {
                    continue;
                }

                let problem = problem_arr.value(i);
                let solution = if solution_arr.is_null(i) {
                    ""
                } else {
                    solution_arr.value(i)
                };
                let thinking = thinking_arr
                    .as_ref()
                    .and_then(|opt| opt.as_ref())
                    .and_then(|arr| {
                        if arr.is_null(i) {
                            None
                        } else {
                            let v = arr.value(i);
                            if v.is_empty() { None } else { Some(v) }
                        }
                    });

                let prompt = problem.to_string();
                let response = match thinking {
                    Some(t) => format!("<think>\n{}\n</think>\n\n{}", t, solution),
                    None => solution.to_string(),
                };

                samples.push(TextSample {
                    text: format!("{}\n\n{}", prompt, response),
                    prompt: Some(prompt),
                });
            }
        }

        Ok(samples)
    }

    /// Load dataset from a Parquet file (legacy, loads as empty).
    #[deprecated(note = "Use from_parquet_tokenized instead")]
    pub fn from_parquet<P: AsRef<Path>>(_path: P) -> Result<Self> {
        Ok(Self::new())
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

    /// Shuffle the dataset.
    pub fn shuffle(&mut self, seed: u64) {
        use rand::SeedableRng;
        use rand::seq::SliceRandom;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        // Build a permutation over indices and apply it to both vecs.
        if self.original_lengths.len() == self.samples.len() && !self.original_lengths.is_empty() {
            let mut indices: Vec<usize> = (0..self.samples.len()).collect();
            indices.shuffle(&mut rng);
            let new_samples: Vec<Sample> =
                indices.iter().map(|&i| self.samples[i].clone()).collect();
            let new_lengths: Vec<usize> =
                indices.iter().map(|&i| self.original_lengths[i]).collect();
            self.samples = new_samples;
            self.original_lengths = new_lengths;
        } else {
            self.samples.shuffle(&mut rng);
        }
    }

    /// Split the dataset into train and validation sets.
    pub fn train_val_split(mut self, val_ratio: f32, seed: u64) -> (Self, Self) {
        self.shuffle(seed);
        let val_size = (self.samples.len() as f32 * val_ratio).round() as usize;
        let val_orig = if self.original_lengths.len() == self.samples.len() {
            self.original_lengths
                .split_off(self.original_lengths.len() - val_size)
        } else {
            Vec::new()
        };
        let val_samples = self.samples.split_off(self.samples.len() - val_size);

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

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Peek at the first record of a JSONL file and return its top-level field names.
///
/// Useful for populating a column-picker UI or validating column config before
/// loading a large dataset. Returns an empty `Vec` if the file is empty or the
/// first record is not a JSON object.
pub fn peek_columns(path: impl AsRef<Path>) -> Result<Vec<String>> {
    let file = File::open(path.as_ref()).map_err(pmetal_core::PMetalError::Io)?;
    let reader = BufReader::new(file);
    for line in reader.lines() {
        let line = line.map_err(pmetal_core::PMetalError::Io)?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Ok(obj) = serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(trimmed)
        {
            return Ok(obj.keys().cloned().collect());
        }
        // First non-empty line is not a JSON object — give up.
        break;
    }
    Ok(Vec::new())
}

mod embedding;
pub use embedding::{EmbeddingDataset, EmbeddingPair, EmbeddingTriplet};

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_sample_creation() {
        let sample = Sample::new(vec![1, 2, 3, 4]);
        assert_eq!(sample.input_ids, vec![1, 2, 3, 4]);
        assert_eq!(sample.attention_mask, vec![1, 1, 1, 1]);
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
        assert!(samples[0].prompt.is_some());
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

    #[test]
    fn test_train_val_split() {
        let samples: Vec<Sample> = (0..100).map(|i| Sample::new(vec![i as u32])).collect();
        let dataset = TrainingDataset::from_samples(samples);

        let (train, val) = dataset.train_val_split(0.2, 42);

        assert_eq!(train.len(), 80);
        assert_eq!(val.len(), 20);
    }

    #[test]
    fn test_parquet_loading() {
        use arrow::array::StringBuilder;
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::arrow_writer::ArrowWriter;
        use std::sync::Arc;

        // Create a temporary Parquet file
        let temp_file = tempfile::NamedTempFile::new().unwrap();
        let path = temp_file.path().to_path_buf();

        // Define schema
        let schema = Arc::new(Schema::new(vec![
            Field::new("text", DataType::Utf8, false),
            Field::new("prompt", DataType::Utf8, true),
        ]));

        // Create data
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

        // Write Parquet file
        let file = File::create(&path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        // Test loading without prompt column
        let samples = TrainingDataset::load_parquet_text(&path, "text", None).unwrap();
        assert_eq!(samples.len(), 3);
        assert_eq!(samples[0].text, "Hello world");
        assert_eq!(samples[1].text, "Another sample");
        assert_eq!(samples[2].text, "Third sample");
        assert!(samples[0].prompt.is_none());

        // Test loading with prompt column
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
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::arrow_writer::ArrowWriter;
        use std::sync::Arc;

        let temp_file = tempfile::NamedTempFile::new().unwrap();
        let path = temp_file.path().to_path_buf();

        let schema = Arc::new(Schema::new(vec![Field::new(
            "content",
            DataType::Utf8,
            false,
        )]));

        let mut builder = StringBuilder::new();
        builder.append_value("Test content");
        let array = Arc::new(builder.finish());

        let batch = RecordBatch::try_new(schema.clone(), vec![array]).unwrap();

        let file = File::create(&path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        // Should fail when requesting non-existent column (and no reasoning columns)
        let result = TrainingDataset::load_parquet_text(&path, "text", None);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("not found"),
            "Error should mention column not found: {}",
            err
        );
    }

    #[test]
    fn test_parquet_reasoning_format() {
        use arrow::array::StringBuilder;
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::arrow_writer::ArrowWriter;
        use std::sync::Arc;

        let temp_file = tempfile::NamedTempFile::new().unwrap();
        let path = temp_file.path().to_path_buf();

        // Create reasoning format Parquet with problem/thinking/solution columns
        let schema = Arc::new(Schema::new(vec![
            Field::new("problem", DataType::Utf8, false),
            Field::new("thinking", DataType::Utf8, true),
            Field::new("solution", DataType::Utf8, false),
        ]));

        let mut problem_builder = StringBuilder::new();
        problem_builder.append_value("What is 2+2?");
        problem_builder.append_value("Explain gravity.");
        let problem_array = Arc::new(problem_builder.finish());

        let mut thinking_builder = StringBuilder::new();
        thinking_builder.append_value("Let me add 2 and 2.");
        thinking_builder.append_null(); // No thinking for second sample
        let thinking_array = Arc::new(thinking_builder.finish());

        let mut solution_builder = StringBuilder::new();
        solution_builder.append_value("4");
        solution_builder.append_value("Gravity is a force of attraction.");
        let solution_array = Arc::new(solution_builder.finish());

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![problem_array, thinking_array, solution_array],
        )
        .unwrap();

        let file = File::create(&path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        // Requesting "text" column should auto-detect reasoning format
        let samples = TrainingDataset::load_parquet_text(&path, "text", None).unwrap();
        assert_eq!(samples.len(), 2);

        // First sample: has thinking
        assert!(samples[0].text.contains("What is 2+2?"));
        assert!(samples[0].text.contains("<think>"));
        assert!(samples[0].text.contains("Let me add 2 and 2."));
        assert!(samples[0].text.contains("</think>"));
        assert!(samples[0].text.contains("4"));
        assert_eq!(samples[0].prompt, Some("What is 2+2?".to_string()));

        // Second sample: no thinking
        assert!(samples[1].text.contains("Explain gravity."));
        assert!(!samples[1].text.contains("<think>"));
        assert!(
            samples[1]
                .text
                .contains("Gravity is a force of attraction.")
        );
        assert_eq!(samples[1].prompt, Some("Explain gravity.".to_string()));
    }
}
