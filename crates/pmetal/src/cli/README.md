# The command line

`cli/mod.rs` holds the `clap` tree (`Cli`, `Commands`); each subcommand's
arguments live in a module of this directory. Every command that runs a job is
the same three steps:

```text
argv --clap--> XxxArgs --From--> pmetal_core::jobs::XxxSpec --normalize--> run(spec)
```

The spec is the single description of the job. The CLI, the TUI, the GUI, the
MCP server and the Python bindings all build the same spec, so a default, a
range or an option list lives once, in `pmetal-core/src/jobs/`.

## Adding or changing a flag

1. Add the field to the spec, with `argv = "--flag"`, a default and its range
   (`#[job(...)]`). The derive then spells it in `to_argv()` and
   `argv_flags()`, and the forms, the MCP tool schemas and the validation pick
   it up.
2. Add the same flag to the command's `XxxArgs` struct and to
   `impl From<XxxArgs> for XxxSpec`. Use `pmetal_core::defaults` constants (or
   the type's own `Default`) for defaults; do not retype a number.
3. Use it in the command's `run`.

`cli/parity.rs` fails when the two sides disagree: a flag the spec cannot
spell, a spec flag the parser does not take, a default that differs, an option
list that differs. `cli/argv_roundtrip.rs` checks that a fully populated spec
comes back as the same spec after going through argv and the parser; those
specs are written out field by field, so a field added to one stops the build
until the round trip covers it.

A flag that exists only on the command line (a path to print to, say) is listed
in `CLI_ONLY` in `parity.rs` with its reason.

## Event log and stopping

`--log-events <file>` (global) writes the run's `JobEvent`s as JSONL. It is
wired to `train`, `distill`, `grpo`, `rlkd` and `preference` through
`commands::with_run_context`; a write that fails is reported when the run ends
(`commands/events.rs`). The other trainers (`pretrain`, `embed-train`,
`train-mtp`, `train-draft`, `train-diffusion`) write no event log, so `main`
refuses the flag for them (`Commands::writes_event_log`) instead of leaving a
caller to wait on an empty file. A command that gains an event log is added to
that method.

Every trainer command stops the same way: Ctrl-C or SIGTERM ends the run after
the step in flight, saves what it has (`commands/stop.rs`) and exits with status
130.

## Documentation

`docs/cli/` has one page per command. `cli/doc_examples.rs` parses every
`pmetal` command in those pages and in `examples/` with the real parser, so a
renamed flag fails a test instead of leaving a page that no longer runs.
