# pmetal mcp

Run PMetal as an MCP server so Claude Desktop, Claude Code, or any other MCP
client can drive training, inference and model management directly.

## Usage

```bash
pmetal mcp
```

The server speaks MCP over stdio: it reads requests on stdin and writes
responses on stdout, so it is started by the client rather than by you. All
logging goes to stderr, which keeps stdout clean for the protocol.

The prebuilt release binary and the Homebrew formula both include this command.
If you build PMetal yourself, `mcp` is opt-in:
`cargo install pmetal --features mcp`.

## Claude Desktop

Add PMetal to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "pmetal": {
      "command": "pmetal",
      "args": ["mcp"]
    }
  }
}
```

Use an absolute path to the binary if `pmetal` is not on the PATH that Claude
Desktop inherits — an app launched from Finder does not see `/opt/homebrew/bin`
or `~/.cargo/bin`. `which pmetal` will tell you where it is.

Restart Claude Desktop, and PMetal's tools appear in the tool picker.

## Claude Code

```bash
claude mcp add pmetal -- pmetal mcp
```

## Tools

55 tools, grouped by what they do.

| Group | Tools |
|-------|-------|
| **Training** | `train`, `pretrain`, `grpo`, `rlkd`, `preference`, `distill`, `embed_train`, `train_diffusion`, `train_draft`, `train_mtp`, `dflash` |
| **Inference** | `generate`, `chat`, `decide`, `start_serve` |
| **Jobs** | `list_jobs`, `job_status`, `job_logs`, `stop_job`, `job_graceful_stop`, `job_save_checkpoint` |
| **Adaptive LR** | `job_set_lr`, `job_reduce_lr`, `job_reset_lr` |
| **Models** | `search_models`, `download_model`, `list_local_models`, `model_info`, `model_fit`, `verify_model`, `delete_local_model`, `inspect_output` |
| **Datasets** | `dataset_analyze`, `dataset_convert`, `dataset_download`, `dataset_filter`, `dataset_merge`, `dataset_prepare`, `dataset_preview`, `dataset_sample`, `dataset_split`, `dataset_template`, `dataset_validate`, `tokenize` |
| **Model ops** | `merge_models`, `quantize`, `fuse_lora`, `pack_experts`, `ollama_create`, `ollama_modelfile` |
| **Evaluation** | `eval_perplexity`, `bench_train`, `bench_gen`, `bench_corpus` |
| **Device** | `device_info` |

Long-running work (training, GRPO, distillation, quantization, serving) starts as a
background job and returns a job ID. Poll it with `job_status` (state, step, loss,
learning rate, tokens per second) and read output with `job_logs`, passing the
`next_since` of the last call to get only what is new. The adaptive-LR tools let a
client steer a run that is already going, and `inspect_output` shows what a finished
run wrote. `generate`, `eval_perplexity`, `bench_train`, `bench_gen`, `bench_corpus`,
`decide` and the `dataset_*` tools run to completion and return their output; while one
runs, a client that asked for progress gets a heartbeat, and cancelling the request stops
the command.

### Parameters

The parameters of a tool that runs a `pmetal` command are the fields of that command's
job spec (the same ones the CLI, the TUI and the GUI use), with the same names, ranges,
options and defaults: `train` takes `output_dir`, `lora_r`, `learning_rate` and so on,
and a parameter left out takes the CLI's default. `tools/list` publishes each as a JSON
Schema with its type, bounds and allowed values. A call that names a parameter the tool
does not take, or gives one the wrong type or an out-of-range value, comes back as an
error result that names the parameter.

Every tool states its behavior hints (`readOnlyHint`, `destructiveHint`,
`idempotentHint`, `openWorldHint`), and tools that return data return it as structured
content as well as text. A tool that fails returns a result with `isError` set, so the
model can read the message and try again; only an unknown tool is a protocol error.

### Resources and prompts

The server publishes `pmetal://device`, `pmetal://jobs`, and for each job
`pmetal://jobs/{job_id}` and `pmetal://jobs/{job_id}/logs`. Three prompts walk through
common workflows: `finetune`, `evaluate_adapter` and `export_for_ollama`.

## Limits and safety

Paths a client gives (outputs, datasets, adapters, local models) are made absolute,
checked for `..` and for symlinks that lead out, and must lie under the server's working
directory, the temporary directory, the model cache, a directory listed in
`PMETAL_MCP_ROOTS` (a `:`-separated list), or a root the client lists (MCP `roots`: a
client that supports them lists its workspace). A model or dataset that is not a path
must be `owner/name` (with `:QUANT` to pick one GGUF quantization). When a launcher that
lists no roots starts the server in `/`, set `PMETAL_MCP_ROOTS` in the server's `env`:

```json
{
  "mcpServers": {
    "pmetal": {
      "command": "pmetal",
      "args": ["mcp"],
      "env": { "PMETAL_MCP_ROOTS": "/path/to/projects:/path/to/models" }
    }
  }
}
```

| Variable | Effect |
|----------|--------|
| `PMETAL_MCP_ROOTS` | Directories tools may read and write, besides the defaults above |
| `PMETAL_MCP_MAX_JOBS` | Jobs that may run at once (default 4); starting another is refused |
| `PMETAL_MCP_TIMEOUT_SECS` | Longest a command that runs to completion may take (default 3600) |
| `PMETAL_MCP_CHAT_TIMEOUT_SECS` | Longest a `chat` or `decide` request to a server may take (default 600) |
| `PMETAL_MCP_ALLOW_REMOTE_BIND` | Set to `1` to let `start_serve` bind to an address other than this machine's own |
| `PMETAL_BIN` | The `pmetal` binary to run (default: the one next to the server, else on `PATH`) |

Jobs run in their own process group. Stopping a job signals the group, and the server
stops every job it started when it exits. A job's output is kept in a bounded buffer
(10,000 lines, 4 MiB, long lines cut), and tokens that appear in output are redacted.
`start_serve` binds to `127.0.0.1` by default, and `chat` and `decide` reach only a
server on this machine, without following redirects.

## Standalone binary

`pmetal-mcp` is also built as its own binary from the `pmetal-mcp` crate, which
is useful if you want an MCP server without the rest of the CLI:

```bash
cargo install --path crates/pmetal-mcp
```

It is not part of the release artifacts — `pmetal mcp` is the supported path.

## See Also

- [pmetal serve](/cli/serve/) — OpenAI- and Anthropic-compatible HTTP server
- [pmetal tui](/cli/tui/) — terminal control center over the same job substrate
