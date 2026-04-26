# CLI Migration Pattern

This directory is where per-subcommand clap argument structs will live once
the `Commands` enum in `src/main.rs` is refactored. This document describes:

- The **current state** (what the code looks like today).
- The **target state** (what each subcommand looks like after migration).
- The **one-time changes** needed in `main.rs`.
- The **substrate contract** every migrated command must satisfy.

The migration is intentionally phased. Each subcommand can be migrated
independently; the binary behaviour is byte-identical to users throughout.

---

## Current state — inline 50-field clap block

Every subcommand in `src/main.rs` looks like:

```rust
// src/main.rs
enum Commands {
    #[cfg(feature = "trainer")]
    Train {
        #[arg(short, long)]
        config: Option<String>,

        #[arg(short, long)]
        model: Option<String>,

        #[arg(short, long)]
        dataset: Option<String>,

        #[arg(short, long, default_value = "./output")]
        output: String,

        #[arg(long, default_value = "16")]
        lora_r: usize,

        // ... ~45 more fields ...
    },
    // ... 30+ more variants
}
```

The handler in `tokio_main` destructures all 50 fields inline, then calls
`orchestrator::run_training(job_config, None, callbacks)`.

Problems:
- Every surface (TUI, GUI, MCP) has to duplicate the full field list.
- `to_argv()` argv string mismatches cause silent bugs in subprocess spawn.
- No validation; arg parsing errors surface as confusing messages.
- `main.rs` is ~3800 LOC and growing with each new flag.

---

## Target state — thin `TrainArgs` + `From<TrainArgs> for TrainSpec`

After migration, `Train` becomes a one-liner in `Commands`:

```rust
// src/main.rs
enum Commands {
    #[cfg(feature = "trainer")]
    Train(crate::cli::train::TrainArgs),
    // ...
}
```

The per-subcommand module lives at `src/cli/train.rs`:

```rust
// src/cli/train.rs
use clap::Args;
use pmetal_core::jobs::TrainSpec;
use pmetal_core::defaults;

/// Thin clap argument struct for `pmetal train`.
///
/// Field names and `#[arg(long = "...")]` strings MUST match the `argv = "..."`
/// attribute in `TrainSpec` (pmetal-core/src/jobs/train.rs).  The spec is the
/// single source of truth for flag names; this struct only adds clap metadata.
#[derive(Args, Debug)]
pub struct TrainArgs {
    /// Path to YAML config (overrides other flags when set).
    #[arg(short, long)]
    pub config: Option<String>,

    /// Model ID or local path.
    #[arg(short = 'm', long = "model")]
    pub model: Option<String>,

    /// Dataset JSONL file or HuggingFace dataset ID.
    #[arg(short = 'd', long = "dataset")]
    pub dataset: Option<String>,

    /// Output directory.
    #[arg(short = 'o', long = "output", default_value = defaults::TRAIN_OUTPUT_DIR)]
    pub output: String,

    /// LoRA rank.
    #[arg(long = "lora-r", default_value_t = defaults::LORA_R)]
    pub lora_r: usize,

    /// LoRA alpha scaling factor.
    #[arg(long = "lora-alpha", default_value_t = defaults::LORA_ALPHA)]
    pub lora_alpha: f32,

    // ... remaining fields ...

    /// Disable adaptive LR.
    #[arg(long = "no-adaptive-lr")]
    pub no_adaptive_lr: bool,
}

impl From<TrainArgs> for TrainSpec {
    fn from(a: TrainArgs) -> Self {
        TrainSpec {
            model: a.model.unwrap_or_default(),
            dataset: a.dataset.unwrap_or_default(),
            output_dir: a.output,
            lora_r: a.lora_r,
            lora_alpha: a.lora_alpha,
            // ...
            no_adaptive_lr: a.no_adaptive_lr,
            config_path: a.config,
            ..TrainSpec::default()
        }
    }
}
```

The handler in `tokio_main` shrinks to:

```rust
Commands::Train(args) => {
    let mut spec = TrainSpec::from(args);
    spec.normalize()
        .map_err(|errs| anyhow::anyhow!("{}", errs[0].message))?;

    // Build extra_callbacks (e.g. JsonlSink if --log-events is set).
    let mut callbacks: Vec<Box<dyn pmetal_core::TrainingCallback>> = Vec::new();
    if let Some(path) = log_events_path {
        use pmetal_core::{JsonlSink, TrainingCallbackToSink};
        let file = std::fs::File::create(&path)?;
        let sink = JsonlSink::new(file);
        callbacks.push(Box::new(TrainingCallbackToSink::new(
            spec.model.clone(),
            sink,
        )));
    }

    orchestrator::run_training(
        orchestrator::TrainingJobConfig::from(spec),
        None,
        callbacks,
    )
    .await?;
}
```

---

## One-time changes needed in `main.rs`

When the migration of a subcommand is ready:

1. Replace the `Variant { field1, field2, ... }` with `Variant(crate::cli::subcommand::SubcmdArgs)`.
2. Move the flag attributes into `src/cli/subcommand.rs` behind `#[derive(Args)]`.
3. Add `impl From<SubcmdArgs> for SubcmdSpec` in the same file.
4. Update the handler arm: destructure `args`, call `SubcmdSpec::from(args)`, call `.normalize()`.
5. Run `cargo test -p pmetal --features trainer -- argv_roundtrip::subcommand_spec_round_trip` to verify the flag names still match.

Do NOT rename any `#[arg(long = "...")]` string — clap flag names are the user-visible contract and are frozen. The spec's `argv = "..."` attribute must match exactly.

---

## Wiring `--log-events` to other subcommands

The `--log-events <path>` global flag is currently wired to `pmetal train` only.
To wire it to another command (e.g. `pmetal distill`):

```rust
// In the Distill arm of `match cli.command`:
let mut extra_callbacks: Vec<Box<dyn pmetal_core::TrainingCallback>> = Vec::new();
if let Some(path) = log_events_path {
    use pmetal_core::{JsonlSink, TrainingCallbackToSink};
    let file = std::fs::File::create(&path)
        .map_err(|e| anyhow::anyhow!("--log-events: {e}"))?;
    let sink = JsonlSink::new(file);
    // job_id identifies this event stream in the consumer (MCP/TUI).
    extra_callbacks.push(Box::new(TrainingCallbackToSink::new(
        format!("distill:{}", teacher_id),
        sink,
    )));
}
// Pass extra_callbacks into run_distillation_cli.
```

The consumer (MCP or TUI subprocess-fallback) reads lines from the file with
`pmetal_core::parse_event(&line)` and dispatches on the `event` tag.

---

## Integration tests

Because `Cli` and `Commands` are private to the binary crate, the round-trip
tests that call `Cli::try_parse_from` live in `src/main.rs::argv_roundtrip`.
Once `Cli` is re-exported from `lib.rs` (a prerequisite for the full migration),
the tests in `tests/cli_argv_roundtrip.rs` can be promoted to true integration
tests.

Run all round-trip tests:

```sh
cargo test -p pmetal --features trainer -- argv_roundtrip
```

---

## Contract: spec is authoritative for flag names

The `argv = "..."` attribute in `pmetal_core::jobs::*` is the single source of
truth for every CLI flag name. If a round-trip test fails:

- The spec's `argv = "..."` is WRONG — fix it to match the CLI.
- Do NOT rename the CLI flag — that breaks human users.

The CLI default values are also authoritative. The spec's `default_*` attribute
should match (see the `--max-seq-len 0` vs core default discussion in the plan).
