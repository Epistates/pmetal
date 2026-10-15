# pmetal-data

Dataset loading and preprocessing for LLM training.

## Overview

This crate provides data loading, preprocessing, and batching utilities optimized for LLM fine-tuning. It supports multiple dataset formats and includes advanced features like sequence packing and chat template application.

## Supported Formats

The format is detected from a row's columns (structured layouts first) or
named explicitly; custom columns override both.

| Format | Description | Example |
|--------|-------------|---------|
| **Messages** | Chat format (content parts, tool calls, `tools`, per-message `weight`) | `{"messages": [...]}` |
| **ShareGPT** | Conversation format | `{"conversations": [...]}` |
| **Alpaca** | Instruction format | `{"instruction": ..., "input": ..., "output": ...}` |
| **Prompt/completion** | Strings or message lists; only the completion trains | `{"prompt": ..., "completion": ...}` |
| **Reasoning** | Problem, thinking, solution | `{"problem": ..., "thinking": ..., "solution": ...}` |
| **Text** | Raw text, trained whole | `{"text": "..."}` |

Rows come from JSONL, JSON, Parquet, Arrow (IPC file or stream) and CSV/TSV
files, or from a directory: a Hugging Face dataset repository's `train` split,
every shard in order (`rows::RowReader`).

## Features

- **Sequence Packing**: First-fit / best-fit decreasing, block-diagonal masks, per-document positions, padding
- **Chat Templates**: Apply model-specific conversation formatting
- **Response Masking**: The loss covers the assistant's turns (or the completion) only, cut from token offsets
- **Resumable Loading**: Epoch order is a function of the seed; `state()` / `restore()` resume mid-epoch; ranks shard equally
- **Length-Grouped Batching, Mixing, Deduplication**: `DataLoader::with_length_grouping`, `TrainingDataset::mix`, `dedup`
- **Streaming Corpora**: `streaming::tokenize_corpus` and a resumable shard reader for pretraining
- **Tokenizer Integration**: HuggingFace tokenizers support

## Usage

### Basic Dataset Loading

```rust,no_run
use pmetal_data::{DataLoader, DataLoaderConfig, DatasetFormat, Tokenizer, TrainingDataset};

fn load(model_dir: &str) -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer = Tokenizer::from_model_dir(model_dir)?;

    let dataset = TrainingDataset::from_jsonl_tokenized(
        "train.jsonl",
        &tokenizer,
        DatasetFormat::Auto,
        2048,
        None, // chat template
        None, // custom column config
    )?;

    let config = DataLoaderConfig {
        batch_size: 4,
        max_seq_len: 2048,
        shuffle: true,
        pad_token_id: tokenizer.pad_token_id().unwrap_or(0),
        ..Default::default()
    };

    let mut loader = DataLoader::new(dataset, config, None);
    while let Some(batch) = loader.next_batch() {
        // batch.input_ids, batch.attention_mask, batch.labels
        let _ = batch.seq_len;
    }
    Ok(())
}
```

### Loading With Options

```rust,no_run
use pmetal_data::chat_templates::detect_chat_template;
use pmetal_data::{AssistantLoss, DatasetLoadOptions, Tokenizer, TrainingDataset};

fn load(model_dir: &str) -> Result<TrainingDataset, Box<dyn std::error::Error>> {
    let tokenizer = Tokenizer::from_model_dir(model_dir)?;
    let template = detect_chat_template(std::path::Path::new(model_dir), "Qwen/Qwen3-0.6B");

    // A file, or a directory of shards; any container; format detected.
    Ok(TrainingDataset::load(
        "data/",
        &tokenizer,
        &DatasetLoadOptions::new(2048)
            .template(Some(&template))
            .assistant_loss(AssistantLoss::AllTurns),
    )?)
}
```

### With Sequence Packing

```rust,no_run
use pmetal_data::{PackerConfig, SequencePacker, TrainingDataset};

fn pack(dataset: &TrainingDataset) -> Result<(), Box<dyn std::error::Error>> {
    let packer = SequencePacker::new(PackerConfig::with_max_length(2048));
    let (batches, stats) = packer.pack_with_stats(dataset.samples())?;
    println!(
        "{} sequences -> {} batches, {:.1}% efficiency",
        stats.num_sequences,
        batches.len(),
        stats.efficiency * 100.0
    );
    Ok(())
}
```

### Chat Template Application

`detect_chat_template` reads the model's real Jinja template from `tokenizer_config.json` and falls
back to a family default when there isn't one.

```rust,no_run
use std::path::Path;

use pmetal_data::chat_templates::{Message, detect_chat_template};

fn format(model_dir: &Path) -> String {
    let template = detect_chat_template(model_dir, "Qwen/Qwen3-0.6B");

    let formatted = template.apply(&[
        Message::user("Hello!"),
        Message::assistant("Hi there!"),
    ]);

    // `response_start` is the byte offset to mask the prompt up to.
    let _prompt_len = formatted.response_start;
    formatted.text
}
```

## Dataset Format Examples

### ShareGPT
```json
{
  "conversations": [
    {"from": "human", "value": "What is 2+2?"},
    {"from": "gpt", "value": "2+2 equals 4."}
  ]
}
```

### Alpaca
```json
{
  "instruction": "Summarize the following text.",
  "input": "Lorem ipsum...",
  "output": "A summary of the text."
}
```

### Messages
```json
{
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi!"}
  ]
}
```

## Modules

| Module | Description |
|--------|-------------|
| `dataset` | Dataset abstractions, loading, mixing and deduplication |
| `rows` | Rows of any dataset container (JSONL, JSON, Parquet, Arrow, CSV, shard directories) |
| `dataloader` | Batching, sharding, resumable epochs |
| `packing` | Sequence packing utilities |
| `streaming` | Pretraining shards: tokenize a corpus, read it back resumably |
| `chat_templates` | Conversation formatting |
| `tokenizer` | Tokenizer integration |

## License

MIT OR Apache-2.0
