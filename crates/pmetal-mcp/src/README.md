# pmetal-mcp — implementation notes

The user-facing tool list, limits and environment variables are in
`docs/cli/mcp.md`. This file is about how the server is put together.

## Where a tool comes from

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

## Paths and references

`policy::PathPolicy` is the only place a path is judged. A spec field is a path
if its descriptor says so (`Path`, `ModelPicker`, `DatasetPicker`) or the tool
lists it in `extra_path_fields`; a `cli_tools` parameter says so in its type.
Defaults are checked as well as what the caller gave, so `./output` cannot
escape the roots either.

## Jobs

`jobs::JobManager` is the MCP face of `pmetal_supervisor::Supervisor`, the
supervisor the terminal UI runs its jobs under as well. It starts a `pmetal`
subcommand in its own process group and owns it until it ends. Output is one
ordered sequence, bounded by lines, bytes and line length, with tokens redacted
and terminal escape sequences removed. Metrics come from the `--log-metrics`
file the server asks the command to write and from `JobEvent` lines on standard
output. The manager caps running jobs and kept finished jobs, `stop_job` returns
once the job is gone, and `shutdown` stops what is left; `run_stdio` calls it
however the server ends.

Commands that run to completion go through `native::Core::run` (a semaphore, so
a few at a time) and `util::run_command` (time limit, bounded output,
cancellation, progress heartbeat).

## Tests

`tests/` drives the server through `McpHandlerExt::handle_request`, the same
JSON-RPC entry point a transport uses, against a shell script standing in for
`pmetal`. The flags the `cli_tools` rows spell, and the argv the spec tools
build, are checked against the CLI's real parser by the `mcp_tools_parse` tests
in the `pmetal` crate (`cargo test -p pmetal --features mcp mcp_tools_parse`).
