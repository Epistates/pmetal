# Feature Flags

Cargo feature flags for enabling optional PMetal functionality.

PMetal uses Cargo feature flags to control which crates and capabilities are compiled.

## Feature Matrix

The default set is `cli`, `dashboard`, `trainer`, `lora`, `merge`, `ane` and `distributed`.
Entries marked `Yes*` are not listed in `default` but arrive through one of those.

| Feature | Default | Crate | Description |
|---------|---------|-------|-------------|
| `cli` | Yes | — | The `pmetal` binary and its CLI-only dependencies |
| `core` | Yes* | `pmetal-core` | Foundation types, configs, traits |
| `gguf` | Yes* | `pmetal-gguf` | GGUF format support |
| `metal` | Yes* | `pmetal-metal` | Metal GPU kernels |
| `hub` | Yes* | `pmetal-hub` | HuggingFace Hub integration |
| `mlx` | Yes* | `pmetal-mlx` | MLX backend |
| `models` | Yes* | `pmetal-models` | LLM architectures |
| `lora` | Yes | `pmetal-lora` | LoRA/QLoRA |
| `trainer` | Yes | `pmetal-trainer` | Training loops (pulls in `data`, `distill`) |
| `data` | Yes* | `pmetal-data` | Dataset loading (via `cli` and `trainer`) |
| `distill` | Yes* | `pmetal-distill` | Knowledge distillation (via `trainer`) |
| `merge` | Yes | `pmetal-merge` | Model merging strategies |
| `distributed` | Yes | `pmetal-distributed` | Distributed training and `pmetal cluster` |
| `ane` | Yes | — | Apple Neural Engine |
| `dashboard` | Yes | — | TUI control center |
| `native-only` | **No** | `pmetal-bridge` | Bridge-only build with no mlx-rs/mlx-sys |
| `lora-metal-fused` | **No** | — | ~2× LoRA training speedup via fused Metal kernels |
| `vocoder` | **No** | `pmetal-vocoder` | BigVGAN neural vocoder |
| `mhc` | **No** | `pmetal-mhc` | Manifold-Constrained Hyper-Connections |
| `serve` | **No** | `pmetal-serve` | OpenAI-compatible inference server (`pmetal serve`) |
| `mcp` | **No** | `pmetal-mcp` | MCP server for Claude Desktop (`pmetal mcp`) |
| `full` | **No** | — | Every sub-crate feature (not `cli`, `serve` or `mcp`) |

`serve` and `mcp` are opt-in so library consumers don't inherit axum and rmcp. The prebuilt release
binary and the Homebrew formula both enable them, so only a build-it-yourself install needs the
flags.

## Usage

```bash
# Default features
cargo install pmetal

# With specific features
cargo install pmetal --features "merge,serve"

# All features
cargo install pmetal --features full

# Minimal build (no ANE)
cargo build --release --no-default-features --features dashboard
```

## As a Library Dependency

```toml
[dependencies]
pmetal = "0.5"                                    # default features
pmetal = { version = "0.5", features = ["full"] } # every sub-crate

# Or depend on the crates you need directly
pmetal-models = "0.5"
pmetal-trainer = "0.5"
pmetal-lora = "0.5"
```

## See Also

- [Installation](/installation/) — Build options
- [Advanced SDK Usage](/sdk/advanced/) — Crate-level API
