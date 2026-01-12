# Contributing to PMetal

Thank you for your interest in contributing to PMetal! This document provides guidelines and information for contributors.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

## Getting Started

### Prerequisites

- **Rust 1.82+**: Install via [rustup](https://rustup.rs/)
- **macOS**: PMetal targets Apple Silicon exclusively
- **Xcode Command Line Tools**: `xcode-select --install`
- **Metal Toolchain**: `xcodebuild -downloadComponent MetalToolchain`

### Setting Up the Development Environment

```bash
# Clone the repository
git clone https://github.com/epistates/pmetal.git
cd pmetal

# Build in debug mode
cargo build

# Run tests
cargo test --all

# Run clippy
cargo clippy --all -- -D warnings
```

## How to Contribute

### Reporting Bugs

Before reporting a bug:
1. Check existing issues to avoid duplicates
2. Collect relevant information:
   - macOS version and Apple Silicon chip (M1, M2, M3, M4, etc.)
   - Rust version (`rustc --version`)
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
3. **Test**: Ensure all tests pass (`cargo test --all`)
4. **Lint**: Run `cargo clippy --all -- -D warnings`
5. **Format**: Run `cargo fmt --all`
6. **Document**: Update documentation for public API changes
7. **Submit**: Open a PR with a clear description

## Coding Guidelines

### Rust Style

- Follow the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Use `rustfmt` for formatting (default configuration)
- Address all `clippy` warnings
- Prefer explicit error handling over `.unwrap()` in library code
- Use descriptive variable names

### Documentation

- Document all public APIs with doc comments
- Include examples in doc comments where helpful
- Update README.md for user-facing changes

### Testing

- Add tests for new functionality
- Maintain existing test coverage
- Use descriptive test names: `test_<function>_<scenario>_<expected>`

### Metal Shaders

When contributing Metal kernels:
- Follow Metal Best Practices Guide
- Use appropriate data types for precision/performance tradeoffs
- Document thread group size assumptions
- Include performance notes where relevant

## Project Structure

```
crates/
├── pmetal-core/       # Core types, traits, configurations
├── pmetal-metal/      # Metal GPU kernels
├── pmetal-mlx/        # MLX backend integration
├── pmetal-models/     # LLM architectures
├── pmetal-lora/       # LoRA/QLoRA implementations
├── pmetal-trainer/    # Training loops
├── pmetal-data/       # Dataset processing
├── pmetal-hub/        # HuggingFace Hub integration
├── pmetal-distill/    # Knowledge distillation
├── pmetal-merge/      # Model merging algorithms
├── pmetal-gguf/       # GGUF format support
├── pmetal-mhc/        # Memory-efficient Hierarchical Context
├── pmetal-distributed/# Distributed training
├── pmetal-vocoder/    # BigVGAN vocoder
└── pmetal-cli/        # Command-line interface
```

### Adding a New Model Architecture

1. Implement the `CausalLMModel` trait in `pmetal-models`
2. Add config parsing in the appropriate module
3. Add architecture detection in `dispatcher.rs`
4. Create LoRA wrapper in `pmetal-lora` if applicable
5. Add tests covering forward pass and generation
6. Update the model support table in README.md

### Adding a New Metal Kernel

1. Create the `.metal` shader in `pmetal-metal/src/kernels/metal/`
2. Add Rust bindings in `pmetal-metal/src/kernels/`
3. Register the kernel in `mod.rs`
4. Add benchmarks if performance-critical
5. Document expected inputs/outputs and thread group configuration

## Performance Considerations

PMetal prioritizes performance. When contributing:

- Profile before and after changes for performance-critical code
- Use `cargo bench` to validate optimizations
- Consider memory usage, especially for large models
- Leverage Metal's parallel capabilities appropriately
- Avoid unnecessary allocations in hot paths

## Licensing

By contributing, you agree that your contributions will be licensed under the same dual MIT/Apache 2.0 license as the project.

## Questions?

Open a GitHub issue for questions about contributing. We're happy to help!
