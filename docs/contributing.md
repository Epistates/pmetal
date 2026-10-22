# Contributing

Guidelines for contributing to PMetal: code style, testing, Metal shaders, and adding new model architectures. The full guide is [`CONTRIBUTING.md`](https://github.com/Epistates/pmetal/blob/main/CONTRIBUTING.md) in the repository.

Thank you for your interest in contributing to PMetal!

## Getting Started

### Prerequisites

- **macOS 14 or later on Apple Silicon**: PMetal targets Apple Silicon exclusively
- **Rust**: install via [rustup](https://rustup.rs/). The repository pins its toolchain in `rust-toolchain.toml`; the minimum supported Rust version for library users is 1.95
- **Xcode Command Line Tools**: `xcode-select --install`
- **Metal Toolchain**: `xcodebuild -downloadComponent MetalToolchain`
- **CMake**: `brew install cmake` (required for the MLX native build)
- **just**: `brew install just` (the project's task runner)
- **bun** (optional, for GUI): `brew install oven-sh/bun/bun`

### Development Setup

```bash
# Install all build dependencies
xcode-select --install
xcodebuild -downloadComponent MetalToolchain
brew install cmake just

git clone https://github.com/epistates/pmetal.git
cd pmetal

cargo build   # Build CLI + TUI
just test     # Workspace tests (single-threaded: MLX's GPU state is not thread-safe)
just lint     # clippy with -D warnings

# GUI development
cd crates/pmetal-gui && bun install && bun tauri dev
```

`just --list` shows every recipe; `just preflight` runs everything CI runs.

## How to Contribute

### Reporting Bugs

Before reporting, collect:

1. macOS version and Apple Silicon chip (`pmetal info` prints both)
2. PMetal version (`pmetal --version`), and the Rust version (`rustc --version`) if you built it
3. Steps to reproduce
4. Expected vs actual behavior
5. Error messages and stack traces

### Pull Requests

1. Fork and create a feature branch from `main`
2. Follow the coding guidelines below
3. Ensure `just test` passes
4. Run `just lint`
5. Run `just fmt` (the GUI's `src-tauri` is a separate workspace that `cargo fmt --all` does not reach)
6. Update docs for public API, flag and behavior changes
7. Open a PR with a clear description

## Coding Guidelines

### Rust Style

- Follow the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Use `rustfmt` (default config)
- Address all `clippy` warnings
- **Avoid `.unwrap()` and `.expect()` in library crates**: return `Result` with `thiserror`, and reserve `expect` for a stated invariant
- Use descriptive variable names

### Job Specs

A command's fields, defaults, ranges and checks live in its job spec under `crates/pmetal-core/src/jobs/`; the CLI, TUI, GUI, MCP server and Python bindings are generated from it. Add or change an option there, then regenerate the GUI's `specs.generated.ts` and the Python stub (see `CONTRIBUTING.md`).

### Testing & Fuzzing

- Add tests for all new functionality
- Use descriptive names: `test_<function>_<scenario>_<expected>`
- The GGUF decoders have fuzz targets (`gguf_reader`, `gguf_dequant`, `gguf_quantize`); CI runs `gguf_reader` on every PR:

```bash
cargo +nightly install cargo-fuzz
cargo +nightly fuzz run gguf_reader -- -max_total_time=60
```

### Adding a New Model Architecture

1. Implement the model in `pmetal-models/src/architectures/`, with its config parsing (see `traits.rs` for `CausalLMModel`)
2. Add the variant, `model_type` detection and the load and forward arms in `dispatcher.rs`
3. LoRA and QLoRA need no per-architecture wrapper: adapters attach to the model's `Linear` layers through `pmetal-lora`'s `AdaptedModel`
4. Add tests covering the forward pass and generation, and a parity check against reference outputs
5. Update the model support tables

### Adding a New Metal Kernel

1. Check that the tensor runtime has no op that does the job (dense attention, RoPE, SwiGLU, LoRA and GEMM run as runtime ops)
2. Create the `.metal` shader in `pmetal-metal/src/kernels/metal/`
3. Add Rust bindings in `pmetal-metal/src/kernels/`
4. Register the kernel in `mod.rs`
5. Add benchmarks if performance-critical
6. Document thread group configuration

## Project Structure

```
crates/
├── pmetal-runtime/     # Tensor runtime: arrays, autodiff, native engines, optimizers, custom Metal kernels
├── pmetal-core/        # Core types, traits, configurations, the job specs
├── pmetal-core-derive/ # Derive macro for the job specs
├── pmetal-metal/       # Metal context, ANE runtime and kernels
├── pmetal-cache/       # KV caches, prefix reuse, speculative-decoding state
├── pmetal-models/      # LLM architectures and the dispatcher
├── pmetal-lora/        # LoRA/QLoRA as a property of Linear
├── pmetal-trainer/     # Training loops (SFT, preference, GRPO, MTP, ...)
├── pmetal-data/        # Dataset processing, chat templates, tokenization
├── pmetal-hub/         # Hugging Face Hub integration
├── pmetal-distill/     # Knowledge distillation (incl. TAID)
├── pmetal-merge/       # Model merging and LoRA fusing
├── pmetal-gguf/        # GGUF reader, writer and quantization
├── pmetal-mhc/         # Manifold-Constrained Hyper-Connections
├── pmetal-distributed/ # Distributed training (mDNS, ring all-reduce)
├── pmetal-vocoder/     # BigVGAN vocoder
├── pmetal-serve/       # OpenAI- and Anthropic-compatible inference server
├── pmetal-supervisor/  # Runs pmetal commands as supervised child processes
├── pmetal-mcp/         # MCP server
├── pmetal-py/          # Python bindings (maturin/PyO3)
├── pmetal/             # Umbrella crate: the CLI, the TUI and re-exports
└── pmetal-gui/         # Desktop GUI (Tauri + Svelte), its own workspace
```

## License

Contributions are licensed under the same dual MIT/Apache-2.0 license as the project.
