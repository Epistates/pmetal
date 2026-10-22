# pmetal-mcp

PMetal as a Model Context Protocol server: Claude Desktop, Claude Code or any
other MCP client can train, run and manage models on a Mac through it.

## Overview

The server speaks MCP over stdio and exposes the PMetal toolkit as tools:
training and the other job types, inference, model and dataset management,
quantization, merging, evaluation and benchmarks. Long work starts as a
background job and returns a job id; the client follows it with `job_status`
and `job_logs` and steers it with `stop_job` and the adaptive-learning-rate
tools.

It is the `mcp` feature of the `pmetal` CLI (`pmetal mcp`; the release binary
and the Homebrew formula include it) and also builds as its own binary,
`pmetal-mcp`. It depends on `pmetal-core` (the job specs its tools are built
from), `pmetal-supervisor` (which runs the jobs), `pmetal-hub` and
`pmetal-metal`, and launches the `pmetal` binary for the work itself.

The user-facing tool list, limits and environment variables are in the
[MCP guide](https://github.com/epistates/pmetal/blob/main/docs/cli/mcp.md).
This file covers setup and how the server is put together.

## Installation

```bash
cargo install pmetal --features mcp     # as `pmetal mcp`
cargo install pmetal-mcp                # as the standalone `pmetal-mcp` binary
```

The standalone binary runs the `pmetal` it finds next to itself, else on
`PATH`; `PMETAL_BIN` names another.

## Usage

An MCP client starts the server, so there is nothing to run by hand. For
Claude Desktop, add it to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "pmetal": { "command": "pmetal", "args": ["mcp"] }
  }
}
```

For Claude Code:

```bash
claude mcp add pmetal -- pmetal mcp
```

As a library, the server is one call; it returns when the client disconnects
or the process is asked to stop, after stopping the jobs it started. Logging
goes to stderr, because stdout is the protocol:

```rust,no_run
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    pmetal_mcp::run_stdio().await
}
```

## Safety

Every path a client gives is made absolute and checked against an allow-list
(the working directory, the temporary directory, the model cache,
`PMETAL_MCP_ROOTS`, the roots the client lists), including symlinks that lead
out. Jobs run in their own process group and the server stops every one it
started when it exits; output is bounded and tokens in it are redacted;
`start_serve` binds to loopback unless `PMETAL_MCP_ALLOW_REMOTE_BIND=1`.

## Feature flags

None.

## Requirements

- Rust 1.95 or later
- macOS 14 or later on Apple Silicon, with a `pmetal` binary for the tools that run jobs

## Implementation notes

### Where a tool comes from

There is one definition of each tool, and it is the one nearest the thing it
runs:

| Source | Module | Tools |
|--------|--------|-------|
| A job spec in `pmetal-core` | `spec_tools` | training, inference, quantize, merge, serve, ... |
| A table row for a CLI command with no spec | `cli_tools` | `dataset_*`, `ollama_*`, `bench_gen`, `bench_corpus` |
| Code in this crate | `native` | the Hub, the cache, jobs, `chat`, `decide` |

`spec_tools` builds a tool's input schema from the spec's `FieldDescriptor`s and
its defaults from `Spec::default()`, deserializes the call into the spec, runs
the spec's `normalize` and runs `to_argv()`. Nothing about a spec is repeated
here, apart from the list of specs, so a field added to or removed from a spec
appears in or leaves the tool. `native` tools declare their arguments as a
struct that is both the schema (`schemars`) and the parse (`serde`, unknown
fields refused).

`schema_check` holds every call to the schema the tool advertised, whichever of
the three it came from.

### Paths and references

`policy::PathPolicy` is the only place a path is judged. A spec field is a path
if its descriptor says so (`Path`, `ModelPicker`, `DatasetPicker`) or the tool
lists it in `extra_path_fields`; a `cli_tools` parameter says so in its type.
Defaults are checked as well as what the caller gave, so `./output` cannot
escape the roots either.

### Jobs

`jobs::JobManager` is the MCP face of `pmetal_supervisor::Supervisor`, the
supervisor the terminal UI and the GUI run their jobs under as well. It starts a
`pmetal` subcommand in its own process group and owns it until it ends. Output
is one ordered sequence, bounded by lines, bytes and line length, with tokens
redacted and terminal escape sequences removed. Metrics come from the
`--log-metrics` file the server asks the command to write and from `JobEvent`
lines on standard output. The manager caps running jobs and kept finished jobs,
`stop_job` returns once the job is gone, and `shutdown` stops what is left;
`run_stdio` calls it however the server ends.

Commands that run to completion go through `native::Core::run` (a semaphore, so
a few at a time) and `util::run_command` (time limit, bounded output,
cancellation, progress heartbeat).

### Tests

`tests/` drives the server through `McpHandlerExt::handle_request`, the same
JSON-RPC entry point a transport uses, against a shell script standing in for
`pmetal`. The flags the `cli_tools` rows spell, and the argv the spec tools
build, are checked against the CLI's real parser by the `mcp_tools_parse` tests
in the `pmetal` crate (`cargo test -p pmetal --features mcp mcp_tools_parse`).

## License

MIT OR Apache-2.0
