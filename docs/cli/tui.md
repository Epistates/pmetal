# pmetal tui

Launch the terminal control center: 21 tabs for monitoring, configuring and running jobs, and chatting with a model.

## Usage

```bash
pmetal tui
pmetal tui --metrics-file training_metrics.jsonl
```

Every job tab is a form built from the same job spec the CLI, the MCP server and the desktop app use, so a field's name, default, range and the command it launches are the spec's. Starting a job shows the exact `pmetal` command first. Jobs run as child processes under the shared job supervisor, each in its own process group, with a bounded, scrollable, secret-redacted log. Leaving the TUI, by `q`, a closed terminal or a signal, stops every job it started.

## Tabs

Listed in tab order, which is also the `Alt+N` order. The tab bar scrolls when the terminal is narrower than the list.

| Tab | Description |
|-----|-------------|
| **System** | GPU/ANE info, Metal feature detection, memory in use against the device's working-set limit, kernel tuning, UltraFusion topology |
| **Models** | Models in the Hugging Face cache and your own directories, with parameters and whether each fits the device; Hub search (`S`), download, fuse |
| **Datasets** | Local datasets (JSONL, Parquet, CSV, Arrow) with line counts, format readiness and a column preview; convert to pmetal JSONL |
| **Tokenize** | Tokenize a text corpus into binary shards for pretraining |
| **Training** | SFT/LoRA/QLoRA training |
| **EmbedTrain** | Train a sentence-embedding (encoder-only) model with contrastive losses |
| **Pretrain** | Full-parameter pretraining from scratch |
| **Distill** | Knowledge distillation (online, offline, progressive) |
| **RLKD** | Reinforcement learning with knowledge distillation |
| **GRPO** | GRPO/DAPO reasoning training with reward functions and sampling parameters |
| **Preference** | DPO, Robust DPO, IPO, hinge, SimPO, ORPO and KTO with LoRA |
| **Monitor** | Live loss curves, learning-rate schedule, throughput and timing of a running or recorded training run (`pmetal dashboard` opens here) |
| **Inference** | Chat with generation settings in a sidebar; settings left alone are not passed to the CLI |
| **DFlash** | Block-diffusion speculative decoding |
| **Serve** | OpenAI-compatible server; shows when it is accepting connections and warns before binding a non-loopback address |
| **Quantize** | GGUF and MLX quantization |
| **Merge** | SLERP, TIES, DARE and linear model merging |
| **Bench** | Training and inference benchmarks, and the workload mode |
| **Eval** | Perplexity evaluation against a dataset |
| **Modelfile** | Modelfile generation for Ollama |
| **Jobs** | Jobs started in this session, and finished runs found on disk, each with its log |

## Keybindings

`?` opens the help overlay, which lists the keys of the tab you are on; the footer shows the most useful ones. While a text field, the model filter or the chat prompt has the keyboard, letters are typed and only the keys below marked "always" act.

| Key | Action |
|-----|--------|
| `Tab` / `Shift+Tab` | Next / previous tab |
| `Ctrl+P` | Go to any tab, type to filter |
| `Alt+1`-`Alt+9` | Jump to one of the first nine tabs |
| `?` | Show / hide help |
| `q` | Quit; asks first when jobs are running, and stops them |
| `Ctrl+C` (always) | Cancel an edit, else stop this tab's running job (on Monitor, the job being followed), else ask to quit |
| `Esc` (always) | Close a dialog or cancel an edit |
| `F2` | Toggle mouse capture; off lets you select text with the terminal |
| `j` / `k`, arrows | Move between fields or list rows |
| `Enter` | Edit, cycle or pick a field's value |
| `Left` / `Right` | Cycle an option backwards / forwards |
| `S` | Review the command and start the job |
| `x` | Stop the running job |
| `L` | Change the learning rate of a running training job |
| `PgUp` / `PgDn`, `End` | Scroll a log; `End` follows it again |
| `Ctrl+U`, `Ctrl+A`, `Ctrl+E` | While editing: clear, start of line, end of line |

On Inference, `Enter` sends, `Ctrl+P` picks a model, `Ctrl+S` focuses the settings, `Ctrl+H` hides them, `Ctrl+B` browses messages (`y` copies one), `Esc` stops a generation and `Ctrl+Q` quits. The Models, Datasets and Jobs tabs add `/` (filter), `R` (rescan), `t`/`s`/`i`/`f` (train, distill, chat, fuse with the selected model), `a` (add a directory) and `c` (convert a dataset).

## Terminal behaviour

The terminal is restored on exit, on a panic and on SIGTERM/SIGHUP. Below 60x14 the TUI asks for a larger window instead of drawing. Colour follows `NO_COLOR`, `COLORTERM` and `TERM`: truecolor, 256 colours, or 16 colours, with text and borders that meet WCAG contrast at each depth. Child output that is not UTF-8 is shown with replacement characters; control sequences in it are removed.

## See Also

- [pmetal dashboard](/cli/dashboard/): The TUI opened on the Monitor tab
- [pmetal train](/cli/train/): CLI training
- [pmetal mcp](/cli/mcp/): The same jobs as MCP tools
