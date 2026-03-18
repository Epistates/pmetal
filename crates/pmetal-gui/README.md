# pmetal-gui

Desktop GUI application for PMetal built with Tauri + Svelte + TailwindCSS.

## Overview

A full-featured desktop application for visual model management, training, and inference on Apple Silicon. The GUI provides a graphical interface for all PMetal operations — from downloading models to configuring LoRA training with live loss metrics.

## Pages

| Page | Description |
|------|-------------|
| **Dashboard** | Overview with device info and training metrics |
| **Models** | Browse, search, download, and manage HuggingFace models |
| **Datasets** | Upload, preview, and manage training datasets |
| **Training** | Configure and launch LoRA/QLoRA/SFT training with live progress |
| **Distillation** | Knowledge distillation from teacher to student models |
| **GRPO** | GRPO/DAPO reasoning training with reward functions |
| **Inference** | Chat interface with generation settings |
| **Merging** | Combine models using multiple merge strategies |
| **Quantize** | GGUF quantization with format selection |
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

```
pmetal-gui/
├── src/                    # Svelte frontend
│   ├── routes/             # SvelteKit pages (10 pages)
│   ├── lib/                # Shared components and utilities
│   └── app.html            # HTML shell
├── src-tauri/              # Rust backend
│   ├── src/
│   │   ├── main.rs         # Tauri entry point
│   │   └── lib.rs          # Tauri commands (IPC bridge)
│   └── Cargo.toml          # Rust dependencies (uses pmetal with "full" features)
├── package.json            # Frontend dependencies
├── svelte.config.js        # Svelte configuration
├── vite.config.ts          # Vite configuration
└── tailwind.config.ts      # TailwindCSS theme
```

The Rust backend imports `pmetal` with all features enabled (`features = ["full"]`) and exposes training, inference, and model operations to the Svelte frontend via Tauri's IPC command system. Training runs in-process with real-time progress updates streamed to the UI.

## License

MIT OR Apache-2.0
