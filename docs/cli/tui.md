# pmetal tui

Launch the full-featured terminal control center with 20 tabs.

Launch the terminal UI — a full control center with 20 tabs for monitoring, configuration, and interaction.

## Usage

```bash
pmetal tui
```

## Tabs

Listed in tab order, which is also the `Alt+N` order.

| Tab | Description |
|-----|-------------|
| **Device** | GPU/ANE info, Metal feature detection, memory gauge, kernel tuning, UltraFusion topology |
| **Models** | Browse cached models, HuggingFace Hub search (`S`), memory fit estimation, download |
| **Datasets** | Scan and preview local datasets (JSONL, Parquet, CSV) with line counts |
| **Tokenize** | Tokenize a text corpus into binary shards for pretraining |
| **Training** | Configure and launch SFT/LoRA/QLoRA training runs with sectioned parameter forms |
| **Embed Train** | Train a sentence-embedding (encoder-only) model with contrastive losses |
| **Pretrain** | Full-parameter pretraining from scratch |
| **Distillation** | Configure knowledge distillation (online, offline, progressive) |
| **RLKD** | Reinforcement learning with knowledge distillation |
| **GRPO** | Configure GRPO/DAPO reasoning training with reward functions and sampling params |
| **Dashboard** | Live loss curves (braille), LR schedule, throughput sparklines, timing breakdown gauges |
| **Inference** | Interactive chat interface with markdown rendering and generation settings sidebar |
| **DFlash** | Block-diffusion speculative decoding |
| **Serve** | OpenAI-compatible server control |
| **Quantize** | GGUF and MLX quantization with bit/method selection |
| **Merge** | SLERP, TIES, DARE and linear model merging |
| **Bench** | Training and inference benchmarking |
| **Eval** | Perplexity evaluation against a dataset |
| **Ollama** | Modelfile generation and Ollama export |
| **Jobs** | Training run history with log viewer, status tracking, and metadata |

## Keybindings

| Key | Action |
|-----|--------|
| `Tab` / `Shift+Tab` | Switch tabs |
| `Alt+1-9` / `Ctrl+1-9` | Jump straight to one of the first nine tabs |
| `L` | Adjust learning rate mid-run |
| `S` | Search HuggingFace Hub (Models tab) |
| `q` | Quit |

## See Also

- [pmetal dashboard](/cli/dashboard/) — Standalone dashboard
- [pmetal train](/cli/train/) — CLI training
