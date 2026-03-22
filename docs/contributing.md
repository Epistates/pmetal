# Contributing

Guidelines for contributing to PMetal — code style, testing, Metal shaders, and adding new model architectures.

Thank you for your interest in contributing to PMetal!

## Getting Started

### Prerequisites

- **Rust 1.86+**: Install via [rustup](https://rustup.rs/)
- **macOS**: PMetal targets Apple Silicon exclusively
- **Xcode Command Line Tools**: `xcode-select --install`
- **Metal Toolchain**: `xcodebuild -downloadComponent MetalToolchain`
- **CMake**: `brew install cmake` (required for MLX native build)
- **bun** (optional, for GUI): `brew install oven-sh/bun/bun`

### Development Setup

```bash
# Install all build dependencies
xcode-select --install
xcodebuild -downloadComponent MetalToolchain
brew install cmake

git clone https://github.com/epistates/pmetal.git
cd pmetal

cargo build           # Build CLI + TUI
cargo test --all      # Run tests
cargo clippy --all -- -D warnings  # Lint

# GUI development
cd crates/pmetal-gui && bun install && bun tauri dev
```

## How to Contribute

### Reporting Bugs

Before reporting, collect:

1. macOS version and Apple Silicon chip (M1–M5)
2. Rust version (`rustc --version`)
3. Steps to reproduce
4. Expected vs actual behavior
5. Error messages and stack traces

### Pull Requests

1. Fork and create a feature branch from `main`
2. Follow the coding guidelines below
3. Ensure `cargo test --all` passes
4. Run `cargo clippy --all -- -D warnings`
5. Run `cargo fmt --all`
6. Update docs for public API changes
7. Open a PR with a clear description

## Coding Guidelines

### Rust Style

- Follow the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Use `rustfmt` (default config)
- Address all `clippy` warnings
- **No `.unwrap()` or `.expect()`** in library crates — return `Result` with `thiserror`
- Use descriptive variable names

### Testing & Fuzzing

- Add tests for all new functionality
- Use descriptive names: `test_<function>_<scenario>_<expected>`
- PMetal enforces continuous fuzzing for data ingest points:

```bash
cargo +nightly install cargo-fuzz
cargo +nightly fuzz run gguf_reader -- -max_total_time=60
```

### Adding a New Model Architecture

1. Implement `CausalLMModel` trait in `pmetal-models`
2. Add config parsing in the appropriate module
3. Add architecture detection in `dispatcher.rs`
4. Create LoRA wrapper in `pmetal-lora` if applicable
5. Add tests covering forward pass and generation
6. Update the model support tables

### Adding a New Metal Kernel

1. Create the `.metal` shader in `pmetal-metal/src/kernels/metal/`
2. Add Rust bindings in `pmetal-metal/src/kernels/`
3. Register the kernel in `mod.rs`
4. Add benchmarks if performance-critical
5. Document thread group configuration

## Project Structure

```
crates/
├── pmetal-core/        # Core types, traits, configurations
├── pmetal-metal/       # Metal GPU kernels + ANE runtime
├── pmetal-mlx/         # MLX backend integration
├── pmetal-models/      # LLM architectures
├── pmetal-lora/        # LoRA/QLoRA implementations
├── pmetal-trainer/     # Training loops (SFT, DPO, GRPO, etc.)
├── pmetal-data/        # Dataset processing, chat templates
├── pmetal-hub/         # HuggingFace Hub integration
├── pmetal-distill/     # Knowledge distillation (incl. TAID)
├── pmetal-merge/       # Model merging algorithms
├── pmetal-gguf/        # GGUF format with imatrix quantization
├── pmetal-mhc/         # Manifold-Constrained Hyper-Connections
├── pmetal-distributed/ # Distributed training
├── pmetal-vocoder/     # BigVGAN vocoder
├── pmetal-serve/       # OpenAI-compatible inference server
├── pmetal-py/          # Python bindings (maturin/PyO3)
├── pmetal-cli/         # Command-line interface + TUI
└── pmetal-gui/         # Desktop GUI (Tauri + Svelte)
```

## License

Contributions are licensed under the same dual MIT/Apache-2.0 license as the project.
