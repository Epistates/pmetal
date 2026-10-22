# Installation

Install PMetal with Homebrew, from a prebuilt binary, from crates.io, or from source.

## Requirements

- **macOS 14 (Sonoma) or later** on Apple Silicon (M1 and later)

For building from source (Homebrew, `cargo install` and a checkout all build):
- **Xcode Command Line Tools**: `xcode-select --install`
- **Rust 1.95 or later**: Install via [rustup](https://rustup.rs/). A checkout pins the toolchain it is tested with in `rust-toolchain.toml`, and rustup installs that on first use
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

## Homebrew

```bash
brew install epistates/tap/pmetal
```

The formula builds the command line from source against the MLX revision PMetal pins, with `serve` and `mcp` enabled, so `pmetal serve` and `pmetal mcp` are available.

## Prebuilt Binaries

Each [release](https://github.com/Epistates/pmetal/releases) carries a code-signed `pmetal-<version>-aarch64-apple-darwin.tar.gz` with the binary and the `mlx.metallib` it was built with (the binary embeds a copy as well), a checksum file, and the desktop app as a DMG. The binary includes `pmetal serve` and `pmetal mcp`.

```bash
gh release download v0.7.0 --repo Epistates/pmetal --pattern 'pmetal-*-aarch64-apple-darwin.tar.gz'
tar xzf pmetal-v0.7.0-aarch64-apple-darwin.tar.gz
sudo install -m 755 pmetal-v0.7.0-aarch64-apple-darwin /usr/local/bin/pmetal
pmetal --version
```

Or download the tarball from the release page in a browser and run the last two commands.

## Install from crates.io

```bash
# CLI, TUI, Neural Engine and distributed training, plus the server and the MCP server
cargo install pmetal --features serve,mcp

# Without the server and the MCP server
cargo install pmetal
```

MLX is linked statically into a release build, so the installed binary does not depend on the build directory. See [Feature Flags](/configuration/feature-flags/) for what each feature enables.

## Build from Source

```bash
git clone https://github.com/epistates/pmetal.git && cd pmetal

# Release build (default features: CLI, TUI, trainer, LoRA, merge, vocoder, ANE, distributed)
cargo build --release --features serve,mcp

# Build without the Neural Engine
cargo build --release --no-default-features --features dashboard,trainer,lora,merge

# Run tests (single-threaded for Metal compatibility)
just test
```

The compiled binary will be at `target/release/pmetal`.

In a checkout, `.cargo/config.toml` links MLX as a shared library (`PMETAL_MLX_STATIC=0`) to keep the many test binaries small. A binary you will move or install needs `PMETAL_MLX_STATIC=1`, which `just build-cli` sets.

## GUI Installation

The release page carries the desktop app as a DMG; it bundles the `pmetal` command line, so it is self-contained.

To build it yourself, the desktop GUI is a Tauri + Svelte application:

```bash
cd crates/pmetal-gui
bun install
bun tauri build
```

The built application bundle will be under `target/release/bundle/` (the GUI shares the repository's `target/` directory).

For development:

```bash
cd crates/pmetal-gui
bun install
bun tauri dev
```

### GUI and the CLI

Every GUI job (training, serving, evaluation, merging and the rest) runs as a `pmetal` process
under the same job supervisor the TUI and the MCP server use; only the chat page generates inside
the app. The release bundle carries the CLI as a sidecar, so a downloaded `.app` is
self-contained.

A GUI you built yourself with a plain `bun tauri build` has no sidecar, and an app launched from Finder inherits launchd's PATH
(`/usr/bin:/bin:/usr/sbin:/sbin`), which does not include Homebrew or cargo. The GUI therefore also
looks for `pmetal` in `~/.cargo/bin`, `/opt/homebrew/bin` and `/usr/local/bin` by absolute path, so
a `cargo install pmetal` or `brew install epistates/tap/pmetal` is found. Set `PMETAL_BIN` to override, as for
every other surface; Settings shows which `pmetal` the app found.

A Hugging Face access token entered in Settings is stored in the macOS Keychain, not in a file, and
is handed to jobs as `HF_TOKEN`.

To reproduce the release bundle locally, including the sidecar:

```bash
just build-gui-bundle
```

## Python SDK

The Python SDK is one abi3 wheel for CPython 3.11 and later, built with maturin and PyO3. It is not published to PyPI yet, so build it from a checkout:

```bash
cd crates/pmetal-py
pip install maturin
maturin develop --profile wheel
```

`just build-wheel` builds the wheel the way a release ships it (MLX linked statically, the command line and the metallib inside the package) into `target/wheels/`. The `wheel` profile unwinds on a Rust panic so Python sees an exception; the plain release profile aborts the interpreter.

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
