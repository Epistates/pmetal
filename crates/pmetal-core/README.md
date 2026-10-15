# pmetal-core

Configuration, job specs, events and shared types for the PMetal framework.

## Overview

Every other crate in the workspace speaks this one's types: the sections of a
training config file, the job specs the CLI, TUI, GUI, MCP server and Python
bindings all build jobs from, the events a job streams back, and the learning
rate schedules.

## Features

- **Configuration**: `ModelConfig`, `LoraConfig`, `TrainingConfig`,
  `DatasetConfig`, and `FullTrainingConfig`, the strict loader for a config
  file (unknown keys are errors; files from earlier releases still load).
- **Validation**: `ConfigValidator` is implemented for every config type and
  reports every problem with its dotted key, instead of clamping or ignoring.
- **Schedules**: `LearningRateScheduler` for constant, linear, cosine, cosine
  with restarts, polynomial and WSD (linear, cosine or `1 - sqrt` cooldown)
  schedules, checked against the reference formulas.
- **Job specs and events**: `jobs::*Spec` (one per `pmetal` subcommand, derived
  with `#[derive(JobSpec)]`) and `JobEvent` with its JSONL codec.
- **Training callbacks and metrics**: `TrainingCallback`, `StepMetrics`,
  `EvalMetrics`.
- **Secret Handling**: `SecretString` for tokens and credentials.

## Usage

```rust
use pmetal_core::prelude::*;

let lora_config = LoraConfig {
    r: 16,
    alpha: 16.0,
    dropout: 0.0,
    target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
    ..Default::default()
};

let training_config = TrainingConfig {
    batch_size: 4,
    learning_rate: 2e-4,
    num_epochs: 1,
    ..Default::default()
};

lora_config.validate().unwrap();
training_config.validate().unwrap();

let loaded = FullTrainingConfig::from_yaml("training:\n  learning_rate: 1e-4\n").unwrap();
assert_eq!(loaded.config.training.learning_rate, 1e-4);
```

## Modules

| Module | Description |
|--------|-------------|
| `config` | Configuration structures for models, LoRA, training and data |
| `scheduler` | Learning rate schedules |
| `jobs` | The job spec for each `pmetal` subcommand |
| `events` | `JobEvent`, sinks and the JSONL codec |
| `traits` | `TrainingCallback`, `ConfigValidator` and shared defaults |
| `types` | Metric and state types |
| `error` | Error types and handling |
| `secrets` | Secure string handling for credentials |

## License

MIT OR Apache-2.0
