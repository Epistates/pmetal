# pmetal-core

Configuration, job specs, events and shared types for the PMetal framework.

## Overview

Every other crate in the workspace speaks this one's types: the sections of a
training config file, the job specs the CLI, TUI, GUI, MCP server and Python
bindings all build jobs from, the events a job streams back, and the learning
rate schedules. It depends on no other workspace crate except its own
`#[derive(JobSpec)]` macro, `pmetal-core-derive`, which it re-exports.

## Features

- **Configuration**: `ModelConfig`, `LoraConfig`, `TrainingConfig`,
  `DatasetConfig`, and `FullTrainingConfig`, the strict loader for a config
  file (unknown keys are errors; files from earlier releases still load).
- **Validation**: `ConfigValidator` is implemented for every config type and
  reports every problem with its dotted key, instead of clamping or ignoring.
- **Schedules**: `LearningRateScheduler` for constant, linear, cosine, cosine
  with restarts, polynomial and WSD (linear, cosine or `1 - sqrt` cooldown)
  schedules, checked against the reference formulas.
- **Job specs**: `jobs::*Spec`, one struct per `pmetal` subcommand, each
  derived with `#[derive(JobSpec)]`. A spec carries the field descriptors a
  form is drawn from (label, group, kind, default, range), the `argv` that
  reproduces it on the command line, and its validation, so a front end never
  holds a default or a flag name of its own. `default_text` and
  `spec_from_text` turn a spec into form text and back. `to_argv` leaves out a
  setting that is at its default: the command's own default applies, and so does
  a `--config` file's value, which a flag at the default's value would override.
- **Job events**: `JobEvent` (started, phase, metric, token, log, progress,
  completed, failed, cancelled) with a JSONL codec, `JobEventSink` and its
  sinks, and `JobStatus` to fold a stream into a state.
- **Training callbacks and metrics**: `TrainingCallback`, `StepMetrics`,
  `EvalMetrics`.
- **Steering a run**: `write_lr_control` hands a running job a new learning
  rate through a control file in its output directory.
- **Secret handling**: `SecretString` for tokens and credentials.

## Installation

```toml
[dependencies]
pmetal-core = "0.7"
```

## Usage

```rust
use pmetal_core::jobs::ServeSpec;
use pmetal_core::prelude::*;

// Configuration, checked: every problem is reported with its key.
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

// A job spec: validated, and turned back into the command line it stands for.
let mut spec = ServeSpec {
    model: "Qwen/Qwen3-4B".into(),
    port: 9000,
    ..Default::default()
};
spec.normalize().unwrap();
let argv = spec.to_argv();
assert_eq!(&argv[..2], ["--model", "Qwen/Qwen3-4B"]);
// Only what differs from the default is passed, so a config file's value can
// apply, except the bind address, which is always stated.
let has = |flag: &str| argv.iter().any(|a| a == flag);
assert!(has("--port") && has("--host") && !has("--request-timeout"));
assert_eq!(ServeSpec::subcommand(), "serve");

// A job's events, one JSON object per line.
let mut line = Vec::new();
pmetal_core::write_event(&mut line, &JobEvent::started("job-1", JobKind::Serve)).unwrap();
let event = pmetal_core::parse_event(std::str::from_utf8(&line).unwrap()).unwrap();
assert_eq!(event.job_id(), "job-1");
```

## Modules

| Module | Description |
|--------|-------------|
| `config` | Configuration structures for models, LoRA, training and data |
| `scheduler` | Learning rate schedules |
| `jobs` | The job spec for each `pmetal` subcommand |
| `fields` | `JobFields`, `FieldDescriptor` and the form-text helpers |
| `events` | `JobEvent`, sinks and the JSONL codec |
| `control` | The learning-rate control file a running job polls |
| `paths` | Absolute, lexically normalised output paths, with the parent created |
| `constants` | Shared numeric tables, such as bytes per weight for each quantization format |
| `traits` | `TrainingCallback`, `ConfigValidator` and shared defaults |
| `types` | Metric and state types |
| `error` | `PMetalError` and `Result` |
| `secrets` | Secure string handling for credentials |

## Feature flags

| Feature | Default | Description |
|---------|---------|-------------|
| `ane` | yes | The `PMetalError::Ane` variant, for the Apple Neural Engine crates that report through this error type |

## Requirements

- Rust 1.95 or later
- No platform requirement: the crate is plain Rust and builds anywhere

## License

MIT OR Apache-2.0
