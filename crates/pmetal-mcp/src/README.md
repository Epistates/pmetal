# pmetal-mcp — implementation notes

The MCP server exposes 53 tools (`#[tool]` methods in `lib.rs`; the user-facing
list is in `docs/cli/mcp.md`). Every tool that starts a `pmetal` subcommand
builds that subcommand's `pmetal_core::jobs::*Spec`, runs `normalize()` for
validation, and spawns `spec.to_argv()`, so a tool's parameters, defaults and
checks are the CLI's. The parity gaps an April 2026 audit recorded here (the
`generate` tool's missing inference flags and the missing `tokenize`, `memory`
and `dflash` tools) are closed.

## Architecture notes

### JobEvent JSONL consumer (landed)

`jobs.rs` now tries `pmetal_core::events::parse_event` on each stdout line before
falling back to the legacy `{"step":N,"loss":F}` flat-JSON parser. This means
MCP works with both old subprocesses (flat metrics JSON) and new ones that emit
`JobEvent` JSONL via `--log-events /dev/stdout`.

The structured path populates `JobMetrics` from `MetricPayload::Step` and
`MetricPayload::Eval`; the legacy path remains for backward compatibility until
all subcommands have been ported to emit `JobEvent` JSONL.

### Ring-buffer cap

`MAX_BUFFER_LINES = 10_000`. When full, the oldest **10%** (1000 lines) are
dropped atomically before inserting the new line. This bounds steady-state memory
at ~10 000 lines per stream regardless of job duration.

### Wire-format stability

`JobStatus` in `jobs.rs` retains `#[serde(tag = "state", rename_all = "snake_case")]`
and its current variant names (`Running`, `Stopping`, `Completed { exit_code }`,
`Failed { exit_code, error }`, `Stopped`). This is intentionally NOT migrated to
`pmetal_core::JobStatus<R>` yet because the wire format differs (core uses
`Running { progress, last_metric }` vs MCP's thin `Running`). A follow-up PR
will migrate after agreeing on the wire contract with existing MCP clients.
