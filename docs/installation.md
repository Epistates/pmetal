# Installation

Install PMetal from prebuilt binaries, crates.io, or build from source.

## Requirements

- **macOS** on Apple Silicon (M1, M2, M3, M4, M5)
- **Xcode Command Line Tools**: `xcode-select --install`

For building from source:
- **Rust 1.86+**: Install via [rustup](https://rustup.rs/)
- **Metal Toolchain**: Required for compiling Metal GPU shaders
- **CMake**: Required for building MLX native dependencies

For GUI development:
- **bun**: `brew install oven-sh/bun/bun`

### Installing Build Dependencies

```bash
# 1. Install Xcode Command Line Tools
xcode-select --install

# 2. Accept the Xcode license (if Xcode is installed)
sudo xcodebuild -license accept

# 3. Install the Metal shader compiler toolchain
xcodebuild -downloadComponent MetalToolchain

# 4. Install CMake (required for MLX native build)
brew install cmake

# 5. Restart your terminal after installation
```

> **Note:** If you installed Xcode but `metal` is not found, try:
> `sudo xcode-select -s /Applications/Xcode.app/Contents/Developer`
>
> After installing Xcode and Metal, a **restart** may be required.

## Prebuilt Binaries

Signed binaries are available on the [Releases](https://github.com/Epistates/pmetal/releases) page:

```bash
curl -fsSL https://github.com/Epistates/pmetal/releases/latest/download/pmetal-aarch64-apple-darwin.tar.gz | tar xz
sudo mv pmetal /usr/local/bin/
pmetal --version
```

## Install from crates.io

```bash
# Default features (CLI + TUI + ANE + Dashboard)
cargo install pmetal

# All features
cargo install pmetal --features full
```

See [Feature Flags](/configuration/feature-flags/) for details on what each feature enables.

## Build from Source

```bash
git clone https://github.com/epistates/pmetal.git && cd pmetal

# Release build (default features: ANE + Dashboard)
cargo build --release

# Build without ANE
cargo build --release --no-default-features --features dashboard

# Run tests (single-threaded for Metal compatibility)
just test
```

The compiled binary will be at `target/release/pmetal`.

## GUI Installation

The desktop GUI is a Tauri + Svelte application:

```bash
cd crates/pmetal-gui
bun install
bun tauri build
```

The built application bundle will be in `target/release/bundle/`.

For development:

```bash
cd crates/pmetal-gui
bun install
bun tauri dev
```

## Python SDK

The Python SDK is built with maturin and PyO3:

```bash
cd crates/pmetal-py
pip install maturin
maturin develop --release
```

Then in Python:

```python
import pmetal
print(pmetal.__version__)
```

## Verify Installation

```bash
# Check version
pmetal --version

# Show device info (GPU, ANE, bandwidth)
pmetal info

# Check memory capacity
pmetal memory

# Search for a model
pmetal search "qwen 0.6b" --detailed
```
