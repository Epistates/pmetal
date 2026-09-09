# pmetal-gui

Desktop GUI application for PMetal built with Tauri + Svelte + TailwindCSS.

## Overview

A full-featured desktop application for visual model management, training, and inference on Apple Silicon. The GUI provides a graphical interface for all PMetal operations — from downloading models to configuring LoRA training with live loss metrics.

## Pages

| Page | Description |
|------|-------------|
| **Dashboard** | Overview with device info and training metrics |
| **Training** | Configure and launch LoRA/QLoRA/SFT training with live progress |
| **GRPO** | GRPO/DAPO reasoning training with reward functions |
| **Distillation** | Knowledge distillation from teacher to student models |
| **Pretrain** | Full-parameter pretraining from scratch |
| **Inference** | Chat interface with generation settings |
| **DFlash** | Block-diffusion speculative decoding |
| **Models** | Browse, search, download, and manage HuggingFace models |
| **Datasets** | Upload, preview, and manage training datasets |
| **Merging** | Combine models using multiple merge strategies |
| **Quantize** | GGUF quantization with format selection |
| **Embed Train** | Sentence-embedding training with contrastive losses |
| **RLKD** | Reinforcement learning with knowledge distillation |
| **Ollama** | Modelfile generation and Ollama export |
| **Serve** | OpenAI-compatible server control |
| **Bench** | Training and inference benchmarking |
| **Eval** | Perplexity evaluation |
| **Jobs** | Run history with log viewer and status tracking |
| **Settings** | Application configuration |

## Tech Stack

- **Tauri 2** — native desktop framework (Rust backend)
- **Svelte** — reactive UI framework
- **TailwindCSS** — utility-first styling
- **Vite** — build tooling
- **bun** — JavaScript runtime and package manager

## Development

```bash
cd crates/pmetal-gui

# Install frontend dependencies
bun install

# Start development server with hot reload
bun tauri dev

# Build production binary
bun tauri build
```

## Architecture

```text
pmetal-gui/
├── src/                    # Svelte frontend
│   ├── routes/             # SvelteKit pages (19 pages)
│   ├── lib/                # Shared components and utilities
│   └── app.html            # HTML shell
├── src-tauri/              # Rust backend
│   ├── src/
│   │   ├── main.rs         # Tauri entry point
│   │   ├── lib.rs          # Builder setup, metallib resolution, command registry
│   │   ├── commands.rs     # Tauri command implementations (IPC bridge)
│   │   └── state.rs        # AppState, config, and run tracking
│   ├── tauri.bundle.conf.json # Release-only: CLI sidecar + metallib resource
│   └── Cargo.toml          # Rust dependencies (uses pmetal with "full" features)
├── package.json            # Frontend dependencies
├── svelte.config.js        # Svelte configuration
├── vite.config.ts          # Vite configuration
└── tailwind.config.ts      # TailwindCSS theme
```

The Rust backend imports `pmetal` with `features = ["full"]` and exposes training, inference, and
model operations to the Svelte frontend via Tauri's IPC command system. Training, inference,
distillation and GRPO run in-process with real-time progress updates streamed to the UI.

The other pages drive the `pmetal` CLI as a subprocess. The release bundle carries it as a Tauri
sidecar; a locally built GUI has none, so it falls back to `~/.cargo/bin`, `/opt/homebrew/bin` and
`/usr/local/bin` by absolute path (an app launched from Finder inherits launchd's `PATH`, which
contains none of those). Set `PMETAL_CLI` to override. `just build-gui-bundle` reproduces the
release bundle locally.

## License

MIT OR Apache-2.0
