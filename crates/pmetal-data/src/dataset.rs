//! Dataset types and loading.

use arrow::array::{Array as ArrowArray, StringArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use pmetal_core::Result;
use serde::Deserialize;
use super::chat_templates::{ChatTemplate, Message};
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatasetFormat {
    /// Simple format: {"text": "..."}
    Simple,
    /// Alpaca format: {"instruction": "...", "input": "...", "output": "..."}
    Alpaca,
    /// ShareGPT format: {"conversations": [{"from": "human", "value": "..."}, ...]}
    ShareGpt,
    /// OpenAI format: {"messages": [{"role": "user", "content": "..."}, ...]}
    OpenAi,
    /// Auto-detect format from first line
    Auto,
}

/// Dataset for training.
#[derive(Clone)]
pub struct TrainingDataset {
    samples: Vec<Sample>,
}

impl TrainingDataset {
    /// Create a new empty dataset.
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
        }
    }

    /// Create a dataset from pre-tokenized samples.
    pub fn from_samples(samples: Vec<Sample>) -> Self {
        Self { samples }
    }

    /// Load and tokenize a dataset from a JSONL file.
    ///
    /// # Arguments
    /// * `path` - Path to the JSONL file
    /// * `tokenizer` - Tokenizer to use
    /// * `format` - Dataset format (or Auto to detect)
    /// * `max_length` - Maximum sequence length
    /// * `template` - Optional chat template for formatting (OpenAI/ShareGPT formats)
    pub fn from_jsonl_tokenized<P: AsRef<Path>>(
        path: P,
        tokenizer: &super::Tokenizer,
        format: DatasetFormat,
        max_length: usize,
        template: Option<&ChatTemplate>,
    ) -> Result<Self> {
        let text_samples = Self::load_jsonl_text(path, format, template)?;
        let mut samples = Vec::with_capacity(text_samples.len());

        for text_sample in text_samples {
            let sample = Self::tokenize_sample(&text_sample, tokenizer, max_length)?;
            samples.push(sample);
        }

        Ok(Self { samples })
    }

    /// Load raw text samples from JSONL without tokenizing.
    pub fn load_jsonl_text<P: AsRef<Path>>(
        path: P,
        format: DatasetFormat,
        template: Option<&ChatTemplate>,
    ) -> Result<Vec<TextSample>> {
        let file = File::open(path.as_ref()).map_err(|e| {
            pmetal_core::PMetalError::Io(std::io::Error::new(
                e.kind(),
                format!("Failed to open dataset file: {}", e),
            ))
        })?;

        let reader = BufReader::new(file);
        let mut samples = Vec::new();
        let mut detected_format = format;

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

            // Auto-detect format on first line
            if line_num == 0 && detected_format == DatasetFormat::Auto {
                detected_format = Self::detect_format(&line)?;
            }

            let sample = Self::parse_line(&line, detected_format, line_num, template)?;
            samples.push(sample);
        }

        Ok(samples)
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
        } else {
            Err(pmetal_core::PMetalError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Could not detect dataset format. Expected 'text', 'instruction', 'conversations', or 'messages' field.",
            )))
        }
    }

    /// Parse a single JSONL line.
    fn parse_line(
        line: &str,
        format: DatasetFormat,
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
            DatasetFormat::Auto => {
                // Should not reach here after detection
                unreachable!("Auto format should be resolved before parsing")
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

        Ok(Self { samples })
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

        // Find column indices
        let text_idx = schema.index_of(text_column).map_err(|_| {
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
        use rand::seq::SliceRandom;
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        self.samples.shuffle(&mut rng);
    }

    /// Split the dataset into train and validation sets.
    pub fn train_val_split(mut self, val_ratio: f32, seed: u64) -> (Self, Self) {
        self.shuffle(seed);
        let val_size = (self.samples.len() as f32 * val_ratio).round() as usize;
        let val_samples = self.samples.split_off(self.samples.len() - val_size);

        (
            self,
            Self {
                samples: val_samples,
            },
        )
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

        let samples = TrainingDataset::load_jsonl_text(file.path(), DatasetFormat::Simple).unwrap();

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

        let samples = TrainingDataset::load_jsonl_text(file.path(), DatasetFormat::Alpaca).unwrap();

        assert_eq!(samples.len(), 1);
        assert!(samples[0].text.contains("Say hello"));
        assert!(samples[0].text.contains("Hello!"));
        assert!(samples[0].prompt.is_some());
    }

    #[test]
    fn test_auto_format_detection() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, r#"{{"text": "Auto detected"}}"#).unwrap();

        let samples = TrainingDataset::load_jsonl_text(file.path(), DatasetFormat::Auto).unwrap();

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

        // Should fail when requesting non-existent column
        let result = TrainingDataset::load_parquet_text(&path, "text", None);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("not found"),
            "Error should mention column not found: {}",
            err
        );
    }
}
