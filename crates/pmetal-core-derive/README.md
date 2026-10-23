# pmetal-core-derive

The `#[derive(JobSpec)]` macro behind PMetal's job specs.

## Overview

Every `pmetal` subcommand has a spec struct in `pmetal-core` (`TrainSpec`,
`ServeSpec`, `MergeSpec`, ...). The derive turns the attributes on a spec's
fields into an implementation of `pmetal_core::JobFields`, so the command
line, the TUI, the GUI, the MCP server and the Python bindings all read one
description of a job: its form fields (label, group, kind, default, range),
the `argv` that reproduces it, and its validation.

You do not depend on this crate directly. `pmetal-core` re-exports the macro as
`pmetal_core::JobSpec`, and the generated code names everything through
`::pmetal_core` and `::std`, so a crate that uses the derive needs
`pmetal-core` and nothing else.

## Features

- **One description per field**: `#[job(label, group, help, default, min, max, kind, argv, ...)]` produces the
  `FieldDescriptor` a form is drawn from, the flag `to_argv` emits and the checks
  `validate_descriptors` runs.
- **Defaults stay out of the argv**: a setting equal to the struct's `Default`
  is not passed (the struct must implement `Default`).
- **Typed argv**: switches (`flag`, `invert`), valued booleans (`valued`),
  comma-joined lists (`csv`) and repeated flags (`repeat`).
- **Compile-time checking**: a key that would silently do nothing (an `invert`
  without `flag`, a `min` on a string, a duplicate key) is a compile error that
  points at the key, and every problem in a struct is reported in one pass.
- **Lint-clean output**: the generated impl carries the struct's generics and
  `where` clause and is marked `#[automatically_derived]`.

## Usage

```rust
use pmetal_core::{JobFields, JobKind, JobSpec};

#[derive(JobSpec)]
#[spec(kind = "Train", subcommand = "train")]
struct MySpec {
    #[job(label = "Model", group = "Model", argv = "--model", kind = "model_picker", required)]
    model: String,

    #[job(
        label = "Learning Rate",
        group = "Training",
        argv = "--learning-rate",
        default_float = 2e-4,
        min = 0.0,
        max = 1.0
    )]
    learning_rate: f64,

    #[job(label = "Tags", group = "Training", argv = "--tag", repeat)]
    tags: Vec<String>,

    #[job(label = "Resume", group = "Training", argv = "--resume", flag)]
    resume: bool,
}

// `to_argv` leaves out a setting that is at its default, so the command's own
// default (or a config file's value) applies; the struct must be `Default`.
impl Default for MySpec {
    fn default() -> Self {
        Self { model: String::new(), learning_rate: 2e-4, tags: Vec::new(), resume: false }
    }
}

let spec = MySpec {
    model: "Qwen/Qwen3-4B".into(),
    learning_rate: 1e-3,
    tags: vec!["a".into(), "b".into()],
    resume: true,
};

assert!(spec.validate_descriptors().is_empty());
assert_eq!(
    spec.to_argv(),
    ["--model", "Qwen/Qwen3-4B", "--learning-rate", "0.001", "--tag", "a", "--tag", "b", "--resume"],
);
assert_eq!(MySpec::subcommand(), "train");
assert!(matches!(MySpec::job_kind(), JobKind::Train));
assert_eq!(MySpec::field_descriptors().len(), 4);

// Out of range: reported against the field, not clamped.
let bad = MySpec { learning_rate: 5.0, ..spec };
assert_eq!(bad.validate_descriptors()[0].field, "learning_rate");
```

The attribute reference (every `#[spec(...)]` and `#[job(...)]` key) is in this
crate's rustdoc.

## Requirements

- Rust 1.95 or later

## License

MIT OR Apache-2.0
