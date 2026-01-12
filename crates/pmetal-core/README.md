# pmetal-core

Core abstractions and types for the PMetal framework.

## Overview

This crate provides the foundational types, traits, and configurations used throughout the PMetal ecosystem. It defines the contracts that all other crates depend on.

## Features

- **Configuration Types**: `ModelConfig`, `LoraConfig`, `TrainingConfig`, `DatasetConfig`
- **Core Traits**: `Model`, `Trainer`, `Quantizer`, `LLMModel`
- **Common Types**: `Dtype`, `Device`, `QuantizationType`
- **Secret Handling**: `SecretString` for secure token/credential storage

## Usage

```rust
use pmetal_core::prelude::*;

// Configure LoRA training
let lora_config = LoraConfig {
    r: 16,
    alpha: 16.0,
    dropout: 0.0,
    target_modules: vec!["q_proj", "v_proj"],
    ..Default::default()
};

// Configure training
let training_config = TrainingConfig {
    batch_size: 4,
    learning_rate: 2e-4,
    epochs: 1,
    ..Default::default()
};
```

## Modules

| Module | Description |
|--------|-------------|
| `config` | Configuration structures for models, training, and data |
| `traits` | Core trait definitions for models and trainers |
| `types` | Common type definitions and enums |
| `error` | Error types and handling |
| `secrets` | Secure string handling for credentials |

## License

MIT OR Apache-2.0
