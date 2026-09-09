# pmetal-data

Dataset loading and preprocessing for LLM training.

## Overview

This crate provides data loading, preprocessing, and batching utilities optimized for LLM fine-tuning. It supports multiple dataset formats and includes advanced features like sequence packing and chat template application.

## Supported Formats

| Format | Description | Example |
|--------|-------------|---------|
| **ShareGPT** | Conversation format | `{"conversations": [...]}` |
| **Alpaca** | Instruction format | `{"instruction": ..., "output": ...}` |
| **Messages** | Chat format | `{"messages": [...]}` |
| **Text** | Raw text | `{"text": "..."}` |

## Features

- **Sequence Packing**: Pack multiple sequences for efficient training
- **Chat Templates**: Apply model-specific conversation formatting
- **Response Masking**: Mask prompt tokens in loss computation
- **Streaming Loading**: Memory-efficient loading of large datasets
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
| `dataset` | Dataset abstractions and loading |
| `dataloader` | Batching and iteration |
| `packing` | Sequence packing utilities |
| `chat_templates` | Conversation formatting |
| `tokenizer` | Tokenizer integration |
| `collator` | Batch collation |

## License

MIT OR Apache-2.0
