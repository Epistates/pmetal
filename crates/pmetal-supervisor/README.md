# pmetal-supervisor

Runs `pmetal` subcommands as supervised child processes: the one place PMetal
jobs are spawned, watched and stopped.

## Overview

The terminal UI, the MCP server and the desktop app all start long jobs
(training, quantization, a server) as `pmetal <subcommand> <argv...>` children.
They used to do it each their own way, and only one of them killed a job's whole
process group. This crate does it once, so all of them behave the same:

- **Process groups.** A job leads its own group. Stopping it signals the group
  (SIGTERM, then SIGKILL after a grace period), so the helpers it spawned go
  with it, and dropping the last `Supervisor` handle kills what is left. No
  child outlives the program that started it.
- **Bounded, ordered output.** Both streams share one ring bounded by lines
  (10,000), bytes (4 MiB) and line length. Lines have tokens and credential
  flags redacted and terminal escape sequences, bidirectional marks and invalid
  UTF-8 cleaned, so they are safe to draw in a terminal and to show to a model.
- **Progress from the job itself.** The files a trainer writes (`--log-events`,
  `--log-metrics`) and `JobEvent` lines on standard output are followed while
  the job runs and folded into `JobMetrics`.
- **Events on a bounded channel.** `Supervisor::subscribe` publishes what
  happens to every job. A subscriber that falls behind is told so and catches
  up from the supervisor's state; a job never waits for a reader.
- **Limits.** `Limits` caps the jobs running at once and the finished jobs kept
  for inspection.

It depends only on `pmetal-core` (the `JobEvent` type). `pmetal-mcp`, the `pmetal`
CLI's terminal UI (its `dashboard` feature), the Python bindings' job API and the
desktop app run their jobs through it.

## Installation

```toml
[dependencies]
pmetal-supervisor = "0.7"
```

## Usage

The supervisor runs whatever executable it is given as `pmetal` (by default,
`resolve_pmetal_binary()`: `PMETAL_BIN`, a `pmetal` beside the running program,
then `PATH`). `echo` stands in for it here:

```rust
use std::time::Duration;

use pmetal_supervisor::{JobRequest, Limits, ProcessStatus, Supervisor};

#[tokio::main]
async fn main() -> Result<(), pmetal_supervisor::Error> {
    let supervisor = Supervisor::new("/bin/echo", Limits::default());
    let id = supervisor.spawn(JobRequest {
        subcommand: "train".into(),
        argv: vec!["--epochs".into(), "1".into()],
        ..JobRequest::default()
    })?;

    let status = supervisor.wait(&id, Duration::from_secs(10)).await?;
    assert!(matches!(status, ProcessStatus::Completed { exit_code: 0 }));

    // The last lines of output, oldest first; `next_since` resumes from here.
    let page = supervisor.logs(&id, 10, None)?;
    assert_eq!(page.lines, ["train --epochs 1"]);

    supervisor.shutdown().await;
    Ok(())
}
```

Following a job as it runs:

```rust,no_run
use pmetal_supervisor::{JobRequest, Limits, Supervisor, resolve_pmetal_binary};

async fn follow() -> Result<(), pmetal_supervisor::Error> {
    let supervisor = Supervisor::new(resolve_pmetal_binary(), Limits::default());
    let mut events = supervisor.subscribe();
    let id = supervisor.spawn(JobRequest {
        subcommand: "info".into(),
        ..JobRequest::default()
    })?;
    while let Ok(event) = events.recv().await {
        if event.job == id {
            println!("{:?}", event.kind);
        }
    }
    supervisor.shutdown().await;
    Ok(())
}
```

## Feature flags

| Feature | Default | Description |
|---------|---------|-------------|
| `schemars` | no | `schemars::JsonSchema` on the summary types, for the MCP server's tool schemas |

## Requirements

- Rust 1.95 or later
- Unix (macOS): process groups and signals

## License

MIT OR Apache-2.0
