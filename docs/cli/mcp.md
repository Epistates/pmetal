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

51 tools, grouped by what they do.

| Group | Tools |
|-------|-------|
| **Training** | `train`, `pretrain`, `grpo`, `rlkd`, `distill`, `embed_train`, `dflash` |
| **Inference** | `generate`, `chat`, `start_serve` |
| **Jobs** | `list_jobs`, `job_status`, `job_logs`, `stop_job`, `job_graceful_stop`, `job_save_checkpoint`, `start_cli_job` |
| **Adaptive LR** | `job_set_lr`, `job_reduce_lr`, `job_reset_lr` |
| **Models** | `search_models`, `download_model`, `list_local_models`, `model_info`, `model_fit`, `estimate_model_memory` |
| **Datasets** | `dataset_analyze`, `dataset_convert`, `dataset_download`, `dataset_filter`, `dataset_merge`, `dataset_prepare`, `dataset_preview`, `dataset_sample`, `dataset_split`, `dataset_template`, `dataset_validate`, `tokenize` |
| **Model ops** | `merge_models`, `quantize`, `fuse_lora`, `pack_experts`, `ollama_create`, `ollama_modelfile` |
| **Evaluation** | `eval_perplexity`, `benchmark`, `bench_train`, `bench_gen`, `bench_corpus` |
| **Device** | `device_info`, `memory` |
| **Escape hatch** | `run_cli` |

Long-running work (training, GRPO, distillation) starts as a background job and
returns a job ID. Poll it with `job_status` and read output with `job_logs`; the
adaptive-LR tools let a client steer a run that is already going.

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
