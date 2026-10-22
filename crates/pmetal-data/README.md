# pmetal-data

Dataset loading and preprocessing for LLM training.

## Overview

This crate provides data loading, preprocessing, and batching utilities optimized for LLM fine-tuning. It supports multiple dataset formats and includes advanced features like sequence packing and chat template application.

`pmetal-trainer`, `pmetal-models`, `pmetal-serve`, the `pmetal` CLI and the Python bindings use it for datasets, tokenizers, chat templates and image preprocessing. It depends on `pmetal-core`, `pmetal-gguf` (GGUF tokenizers) and `pmetal-runtime` (batches are runtime arrays).

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
- **Tokenizer Integration**: HuggingFace tokenizers, including the tokenizer a GGUF-only checkpoint carries in its metadata (`gguf_tokenizer::build_tokenizer`)
- **Chat templates**: a model's own Jinja template, executed as shipped (`jinja_chat`), with built-in renderings for base models; `ReasoningSplitter` and `StreamFormatter` separate a thinking model's reasoning from its answer in a token stream
- **Image preprocessing for vision models**: fixed-size and SigLIP, Gemma 4 patch-budget, Llama 3.2 Vision tiled and Qwen 3.5 image and video processors, on a Pillow-exact resampler (`pillow_resample`) so a model sees the pixels its reference processor produces
- **Sampling presets**: stop tokens and the sampling defaults a model family's card recommends (`inference_config`)

## Installation

```toml
[dependencies]
pmetal-data = "0.7"
```

## Usage

### Packing and batching, no tokenizer needed

A dataset is a list of token-id samples; the packer first-fit packs them into fixed-length rows
with block-diagonal masks, and the loader batches them.

```rust
use pmetal_data::{DataLoader, DataLoaderConfig, PackerConfig, Sample, SequencePacker, TrainingDataset};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Eight samples of 100, 200, ... 800 tokens.
    let samples: Vec<Sample> = (1..=8u32).map(|n| Sample::new((0..n * 100).collect())).collect();
    let dataset = TrainingDataset::from_samples(samples);

    let packer = SequencePacker::new(PackerConfig::with_max_length(1024));
    let (rows, stats) = packer.pack_with_stats(dataset.samples())?;
    assert!(rows.len() < dataset.len());
    assert_eq!(stats.num_sequences, 8);

    let config = DataLoaderConfig {
        batch_size: 4,
        max_seq_len: 1024,
        shuffle: false,
        ..Default::default()
    };
    let mut loader = DataLoader::new(dataset, config, None);
    let mut batches = 0;
    while let Some(batch) = loader.next_batch() {
        assert_eq!(batch.input_ids.shape(), [batch.batch_size as i32, batch.seq_len as i32]);
        batches += 1;
    }
    assert_eq!(batches, 2);
    Ok(())
}
```

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

`detect_chat_template` reads the model's own Jinja template (`chat_template.jinja`,
`tokenizer_config.json`, or a GGUF's `tokenizer.chat_template`) and renders every conversation
with it, as `apply_chat_template` does, so a model is trained on the text it is served. A
checkpoint with no template (a base model) is rendered in the conventional format of its family.

```rust,no_run
use std::path::Path;

use pmetal_data::Tokenizer;
use pmetal_data::chat_templates::{Message, TrainingSampleBuilder, detect_chat_template};

fn format(model_dir: &Path) -> pmetal_core::Result<(Vec<u32>, Vec<i64>)> {
    let template = detect_chat_template(model_dir, "Qwen/Qwen3-0.6B");
    let tokenizer = Tokenizer::from_model_dir(model_dir)?;

    // Every assistant turn is a response span; the labels are -100 everywhere else.
    TrainingSampleBuilder::new(template).build_tokenized(
        &[Message::user("Hello!"), Message::assistant("Hi there!")],
        &tokenizer,
        2048,
    )
}
```

A tokenizer also reports the span of the text each token covers
(`Tokenizer::encode_with_offsets`) and encodes a dataset in parallel (`encode_batch`).

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
| `chat_templates`, `jinja_chat` | Conversation formatting: template detection, response masks, upstream Jinja execution |
| `tokenizer`, `gguf_tokenizer` | Tokenizer integration, and rebuilding one from GGUF metadata |
| `stream_format` | Splitting a reasoning channel from the answer in a token stream |
| `inference_config` | Stop tokens and per-family sampling presets |
| `image_processing`, `qwen_vl_processing`, `pillow_resample` | Image and video preprocessing for vision models |

## Feature flags

None.

## Requirements

- Rust 1.95 or later
- Apple Silicon Mac: batches are `pmetal-runtime` arrays

## License

MIT OR Apache-2.0
