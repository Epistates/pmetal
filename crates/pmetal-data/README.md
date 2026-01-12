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

```rust
use pmetal_data::{Dataset, DataLoader};

// Load dataset
let dataset = Dataset::from_jsonl("train.jsonl")?;

// Create dataloader
let loader = DataLoader::new(dataset, batch_size: 4, shuffle: true);

for batch in loader {
    // batch.input_ids, batch.attention_mask, batch.labels
}
```

### With Sequence Packing

```rust
use pmetal_data::{Dataset, SequencePacker};

let dataset = Dataset::from_jsonl("train.jsonl")?;

// Pack sequences for efficient training
let packed = SequencePacker::pack(&dataset, max_length: 2048)?;
// Reports: "Packing: 1000 sequences → 850 batches, 99.5% efficiency"
```

### Chat Template Application

```rust
use pmetal_data::ChatTemplate;

let template = ChatTemplate::from_tokenizer(&tokenizer)?;

let formatted = template.apply(&[
    Message::user("Hello!"),
    Message::assistant("Hi there!"),
])?;
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
