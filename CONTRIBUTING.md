# Contributing to PMetal

Thank you for your interest in contributing to PMetal! This document provides guidelines and information for contributors.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

## Getting Started

### Prerequisites

- **macOS 14 or later on Apple Silicon**: PMetal targets Apple Silicon exclusively
- **Rust**: install via [rustup](https://rustup.rs/). The repository pins its toolchain in `rust-toolchain.toml` (rustup installs it on first use); the minimum supported Rust version for library users is 1.95
- **Xcode Command Line Tools**: `xcode-select --install`
- **Metal Toolchain**: `xcodebuild -downloadComponent MetalToolchain`
- **CMake**: `brew install cmake` (builds MLX)
- **[just](https://github.com/casey/just)**: `brew install just` (the project's task runner)
- **bun** (optional, for GUI development): `brew install oven-sh/bun/bun`

### Setting Up the Development Environment

```bash
# Clone the repository
git clone https://github.com/epistates/pmetal.git
cd pmetal

# Build CLI + TUI
cargo build

# Run the workspace tests (single-threaded: MLX's GPU state is not thread-safe)
just test

# Lint
just lint

# GUI development (requires bun)
cd crates/pmetal-gui
bun install
bun tauri dev
```

The first build compiles MLX, which takes several minutes and about 1 GB of disk; later builds reuse it. `just --list` shows every recipe, and `just preflight` runs everything CI runs (formatting, clippy with default, all and minimal features, rustdoc, tests in debug and release, the GUI's checks, version and lockfile checks, and `cargo deny`).

## How to Contribute

### Reporting Bugs

Before reporting a bug:
1. Check existing issues to avoid duplicates
2. Collect relevant information:
   - macOS version and Apple Silicon chip (`pmetal info` prints both)
   - PMetal version (`pmetal --version`) and Rust version (`rustc --version`) if you built it
   - Steps to reproduce
   - Expected vs actual behavior
   - Error messages and stack traces

### Suggesting Features

Feature requests are welcome! Please:
1. Describe the use case and motivation
2. Explain the proposed solution
3. Consider alternatives you've thought about

### Pull Requests

1. **Fork and branch**: Create a feature branch from `main`
2. **Make changes**: Follow the coding guidelines below
3. **Test**: Ensure all tests pass (`just test`)
4. **Lint**: Run `just lint` (clippy with `-D warnings`)
5. **Format**: Run `just fmt` (the GUI's `src-tauri` is its own workspace, so `cargo fmt --all` alone misses it)
6. **Document**: Update documentation for public API, flag and behavior changes
7. **Submit**: Open a PR with a clear description

## Coding Guidelines

### Rust Style

- Follow the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Use `rustfmt` for formatting (default configuration)
- Address all `clippy` warnings (CI runs `-D warnings`)
- **Avoid `unwrap()` and `expect()` in library crates.** Return `Result` and use `thiserror` for meaningful error propagation; reserve `expect` for an invariant, with a message that states it
- Unsafe code is denied by a workspace lint; the few FFI modules that need it opt in explicitly
- Use descriptive variable names

### Job specs are the single source of options

A command's fields, defaults, ranges and checks live in its job spec under `crates/pmetal-core/src/jobs/`. The CLI, TUI, GUI, MCP server and Python bindings are generated from those specs, so add or change an option there, not in each surface. Then regenerate what is derived from the specs: the GUI's `src/lib/specs.generated.ts` (`UPDATE_SPEC_TYPES=1 cargo test --manifest-path crates/pmetal-gui/src-tauri/Cargo.toml spec_types`) and the Python stub (`python stub.py` in `crates/pmetal-py`). Removing a flag means removing it from the spec as well, or it stays a live control in the GUI and the MCP server.

### Documentation

- Document all public APIs with doc comments
- Include examples in doc comments where helpful
- Update `README.md` and `docs/` for user-facing changes
- Every code block in a crate's `README.md` is compiled as a doctest (the READMEs are the crates' rustdoc front pages), so editing one can fail `cargo test`

### Testing & Fuzzing

- Add unit and integration tests for all new functionality.
- Maintain existing test coverage.
- Use descriptive test names: `test_<function>_<scenario>_<expected>`
- Tests that touch the GPU run single-threaded (`just test` does this).
- **Fuzzing:** the GGUF decoders have fuzz targets (`gguf_reader`, `gguf_dequant`, `gguf_quantize`), and CI runs `gguf_reader` on every PR.
  - To run fuzz tests locally, install the nightly toolchain and `cargo-fuzz`:
    ```bash
    cargo +nightly install cargo-fuzz
    cargo +nightly fuzz run gguf_reader -- -max_total_time=60
    ```
  - Any panics discovered by the fuzzer must be resolved before a PR will be accepted.

### Metal Shaders

When contributing Metal kernels:
- Check first that the tensor runtime has no op that does the job: dense attention, RoPE, SwiGLU, LoRA and GEMM run as runtime ops, and hand-written kernels for them measured slower and were removed
- Follow Metal Best Practices Guide
- Use appropriate data types for precision/performance tradeoffs
- Document thread group size assumptions
- Include performance notes where relevant

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

### Adding a New Model Architecture

1. Implement the model in `pmetal-models/src/architectures/`, with its config parsing (see `traits.rs` for `CausalLMModel`)
2. Add the variant, `model_type` detection and the load and forward arms in `dispatcher.rs`
3. LoRA and QLoRA need no per-architecture wrapper: adapters attach to the model's `Linear` layers through `pmetal-lora`'s `AdaptedModel`, so check that the module's parameter names and gradient checkpointing work with it
4. Add tests: forward pass and generation, and a parity check against reference outputs for a small config (the existing architectures are checked against `transformers`)
5. Update the model support tables (`README.md`, `docs/models/supported.md`, `crates/pmetal-models/README.md`)

### Adding a New Metal Kernel

1. Create the `.metal` shader in `pmetal-metal/src/kernels/metal/`
2. Add Rust bindings in `pmetal-metal/src/kernels/`
3. Register the kernel in `mod.rs`
4. Add benchmarks if performance-critical
5. Document expected inputs/outputs and thread group configuration

## Performance Considerations

PMetal prioritizes performance. When contributing:

- Profile before and after changes for performance-critical code
- Use `cargo bench` to validate optimizations, and `pmetal bench-workload` for an end-to-end check
- Benchmarks record the machine's load average; take timings on a quiet machine
- Consider memory usage, especially for large models
- Leverage Metal's parallel capabilities appropriately
- Avoid unnecessary allocations in hot paths

## Dependencies

Every dependency is declared once, in the root `[workspace.dependencies]`, and kept at its latest release, but not its newest hour:

- **Wait a week for a new release.** Most malicious releases are caught and pulled within days, so a version younger than seven days is not taken unless it fixes a security advisory. Bun enforces this when it resolves (`minimumReleaseAge` in `crates/pmetal-gui/bunfig.toml`), as does pip in CI (`PIP_UPLOADED_PRIOR_TO`); for crates, check with `just audit-deps`.
- **No install scripts.** `package.json` trusts no dependency, so bun runs no package's install scripts. A package that needs one is a new dependency to discuss in the pull request.
- **Lockfiles are the install.** CI installs with `bun install --frozen-lockfile` and builds against the committed `Cargo.lock` files; GitHub Actions are pinned to a commit, with the release in a comment.
- **Check before you push a dependency change.** `just audit-deps` checks every pinned crate and npm package against OSV's malware and advisory records and lists the versions younger than a week; `just deny` checks advisories, licenses and sources for the crates the macOS build compiles. `just deps-check` reports what is behind.

## Licensing

By contributing, you agree that your contributions will be licensed under the same dual MIT/Apache 2.0 license as the project. New dependencies must have a permissive license (`just deny` checks this against `deny.toml`), and third-party code you adapt needs an attribution in `THIRD_PARTY_NOTICES`.

## Questions?

Open a GitHub issue for questions about contributing. We're happy to help!
