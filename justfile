# PMetal Development Justfile
# Run `just --list` to see all available recipes

set shell := ["zsh", "-cu"]

# Workspace version (single source of truth)
version := `grep '^version = ' Cargo.toml | head -1 | sed 's/version = "//;s/"//'`

# Default recipe - show help
default:
    @just --list

# ─── Pre-publish gate ───────────────────────────────────────────────
# Runs every check that CI performs. If this passes, publish is safe.

# Full pre-publish validation (mirrors CI + release pipelines)
preflight: fmt-check check-workspace-deps lint lint-all-features lint-min-features test test-release check-gui lint-gui check-version check-lockfile deny
    @echo ""
    @echo "All preflight checks passed -- safe to publish {{ version }}"

# ─── Formatting ─────────────────────────────────────────────────────

# Format code.
#
# pmetal-gui/src-tauri is `exclude`d from the workspace, so `--all` does not
# reach it. It needs its own invocation or it drifts unformatted.
fmt:
    cargo fmt --all
    cargo fmt --manifest-path crates/pmetal-gui/src-tauri/Cargo.toml

# Check formatting without changing files (CI: check job)
fmt-check:
    cargo fmt --all -- --check
    cargo fmt --manifest-path crates/pmetal-gui/src-tauri/Cargo.toml -- --check

# ─── Linting ────────────────────────────────────────────────────────

# Clippy with default features, excluding pmetal-py (CI: check job)
lint:
    cargo clippy --workspace --all-targets --exclude pmetal-py -- -D warnings

# Clippy with ALL features enabled (catches cfg-gated field mismatches)
lint-all-features:
    cargo clippy --workspace --all-targets --all-features -- -D warnings

# Clippy on the smallest CLI build (`cli` alone: no trainer, LoRA or ANE), and
# on pmetal-trainer built alone, without the `distributed` feature the
# workspace build turns on for it (as the Python bindings build it). Nothing
# else compiles the paths those features gate off, so unused-code warnings
# there went unseen.
lint-min-features:
    cargo clippy -p pmetal --all-targets --no-default-features --features cli -- -D warnings
    cargo clippy -p pmetal-trainer --all-targets -- -D warnings

# ─── Building ───────────────────────────────────────────────────────

# Build the project in release mode (CI: check job)
build:
    cargo build --workspace --exclude pmetal-py --release

# Build in debug mode
build-debug:
    cargo build --workspace --exclude pmetal-py

# Check compilation without building
check:
    cargo check --workspace --exclude pmetal-py

# Build CLI binary (mirrors release CI: feature set + self-contained MLX)
build-cli:
    PMETAL_MLX_STATIC=1 cargo build --release -p pmetal --features serve,mcp

# Typecheck and build the GUI frontend.
#
# The Rust side needs this first: `tauri::generate_context!` embeds
# `frontendDist` at compile time and panics when the directory is missing, so
# `check-gui` used to pass only on a checkout where someone had already built
# the frontend.
gui-frontend:
    cd crates/pmetal-gui && bun install --frozen-lockfile && bun run check && bun run build

# Check GUI compiles (mirrors release CI -- catches cfg mismatches)
check-gui: gui-frontend
    cargo check --manifest-path crates/pmetal-gui/src-tauri/Cargo.toml

# Clippy for the GUI crate, which `--workspace` does not reach (see `fmt`).
lint-gui: gui-frontend
    cargo clippy --manifest-path crates/pmetal-gui/src-tauri/Cargo.toml --all-targets -- -D warnings

# Stage the CLI sidecar + metallib the release bundle expects.
#
# `bun tauri dev` and a plain `bun tauri build` do NOT need this: externalBin
# lives in tauri.bundle.conf.json, which is only merged for release bundles.
# Run this before `just build-gui-bundle`.
stage-gui-sidecar: build-cli
    #!/usr/bin/env zsh
    set -euo pipefail
    SIDECAR_DIR=crates/pmetal-gui/src-tauri/binaries
    TRIPLE=$(rustc -vV | sed -n 's/^host: //p')
    mkdir -p "$SIDECAR_DIR"
    cp target/release/pmetal "$SIDECAR_DIR/pmetal-$TRIPLE"
    # The metallib the CLI itself loads, checked against the one it embeds.
    INFO=$(env -u PMETAL_METALLIB_PATH target/release/pmetal info --json)
    METALLIB=$(jq -r '.mlx.metallib // empty' <<<"$INFO")
    EXPECTED=$(jq -r '.mlx.metallib_sha256 // empty' <<<"$INFO")
    if [[ -z "$METALLIB" || -z "$EXPECTED" || ! -f "$METALLIB" ]]; then
        echo "FAIL: the CLI reports no mlx.metallib"
        exit 1
    fi
    if [[ $(shasum -a 256 "$METALLIB" | cut -d' ' -f1) != "$EXPECTED" ]]; then
        echo "FAIL: $METALLIB is not the mlx.metallib the CLI embeds ($EXPECTED)"
        exit 1
    fi
    cp "$METALLIB" "$SIDECAR_DIR/mlx.metallib"
    echo "Staged sidecar for $TRIPLE"

# Build the release GUI bundle the way release CI does (sidecar included)
build-gui-bundle: stage-gui-sidecar
    cd crates/pmetal-gui && bun run tauri build -c src-tauri/tauri.bundle.conf.json

# ─── Testing ────────────────────────────────────────────────────────

# Run all workspace tests, single-threaded for Metal GPU safety (CI: test job)
test:
    cargo test --workspace --exclude pmetal-py -- --test-threads=1

# Run all workspace tests in release mode (CI: release test job)
test-release:
    cargo test --workspace --exclude pmetal-py --release -- --test-threads=1

# Run tests for a specific package
test-pkg pkg:
    cargo test --package {{ pkg }} -- --test-threads=1

# Run tests with output shown
test-verbose:
    cargo test --workspace --exclude pmetal-py -- --test-threads=1 --nocapture

# Run only unit tests (no doc tests)
test-unit:
    cargo test --workspace --exclude pmetal-py --lib -- --test-threads=1

# Run specific test by name
test-name name:
    cargo test {{ name }} -- --test-threads=1 --nocapture

# ─── Version & Release ──────────────────────────────────────────────

# Verify all workspace crate versions are consistent
check-version:
    #!/usr/bin/env zsh
    set -euo pipefail
    WS_VER="{{ version }}"
    echo "Workspace version: $WS_VER"
    ERRORS=0
    # Check pmetal-py uses the workspace version
    if ! grep -q '^version.workspace = true$' crates/pmetal-py/Cargo.toml; then
        echo "FAIL: pmetal-py does not use version.workspace = true"
        ERRORS=$((ERRORS + 1))
    fi
    # Check GUI versions
    GUI_VER=$(grep '^version = ' crates/pmetal-gui/src-tauri/Cargo.toml | sed 's/version = "//;s/"//')
    if [[ "$GUI_VER" != "$WS_VER" ]]; then
        echo "FAIL: pmetal-gui Cargo.toml version = \"$GUI_VER\" (expected \"$WS_VER\")"
        ERRORS=$((ERRORS + 1))
    fi
    GUI_PACKAGE_VER=$(grep '"version":' crates/pmetal-gui/package.json | sed 's/.*"version": "//;s/".*//')
    if [[ "$GUI_PACKAGE_VER" != "$WS_VER" ]]; then
        echo "FAIL: pmetal-gui package.json version = \"$GUI_PACKAGE_VER\" (expected \"$WS_VER\")"
        ERRORS=$((ERRORS + 1))
    fi
    GUI_TAURI_CONF_VER=$(grep '"version":' crates/pmetal-gui/src-tauri/tauri.conf.json | sed 's/.*"version": "//;s/".*//')
    if [[ "$GUI_TAURI_CONF_VER" != "$WS_VER" ]]; then
        echo "FAIL: pmetal-gui tauri.conf.json version = \"$GUI_TAURI_CONF_VER\" (expected \"$WS_VER\")"
        ERRORS=$((ERRORS + 1))
    fi
    GUI_LAYOUT_VER=$(grep 'let appVersion' crates/pmetal-gui/src/routes/+layout.svelte | sed "s/.*'\([^']*\)'.*/\1/")
    if [[ "$GUI_LAYOUT_VER" != "$WS_VER" ]]; then
        echo "FAIL: pmetal-gui layout appVersion = \"$GUI_LAYOUT_VER\" (expected \"$WS_VER\")"
        ERRORS=$((ERRORS + 1))
    fi
    # Check CHANGELOG has entry
    if ! grep -q "\[$WS_VER\]" CHANGELOG.md; then
        echo "FAIL: CHANGELOG.md missing entry for [$WS_VER]"
        ERRORS=$((ERRORS + 1))
    fi
    if [[ $ERRORS -gt 0 ]]; then
        echo "FAIL: $ERRORS version inconsistencies found"
        exit 1
    fi
    echo "All versions consistent"

# Verify Cargo.lock satisfies the manifests without needing changes.
#
# `cargo update --locked` is NOT this check: `--locked` forbids writing the
# lockfile and `cargo update` exists to write it, so that command fails the
# moment any dependency anywhere publishes a new version. It never passes for
# long, which is how a red `preflight` went unnoticed. `cargo metadata --locked`
# resolves the graph and fails only if Cargo.lock would have to change.
# Yanked crates are `just audit`'s job, not this one.
check-lockfile:
    cargo metadata --locked --format-version 1 > /dev/null || (echo "FAIL: Cargo.lock is out of date -- run cargo update --workspace" && exit 1)
    @echo "Cargo.lock satisfies the manifests"

# Bump version across all crates (updates workspace, gui, changelog stub)
bump new_version:
    #!/usr/bin/env zsh
    set -euo pipefail
    OLD="{{ version }}"
    NEW="{{ new_version }}"
    echo "Bumping $OLD -> $NEW"
    # Workspace Cargo.toml
    sed -i '' "s/version = \"$OLD\"/version = \"$NEW\"/g" Cargo.toml
    # GUI
    sed -i '' "s/version = \"$OLD\"/version = \"$NEW\"/" crates/pmetal-gui/src-tauri/Cargo.toml
    sed -i '' "s/\"version\": \"$OLD\"/\"version\": \"$NEW\"/" crates/pmetal-gui/package.json
    sed -i '' "s/\"version\": \"$OLD\"/\"version\": \"$NEW\"/" crates/pmetal-gui/src-tauri/tauri.conf.json
    sed -i '' "s/appVersion = \$state('$OLD')/appVersion = \$state('$NEW')/" crates/pmetal-gui/src/routes/+layout.svelte
    # Update lockfile
    cargo update --workspace
    echo "Bumped to $NEW -- update CHANGELOG.md before publishing"

# ─── Dependencies ───────────────────────────────────────────────────

# Update all dependencies
update:
    cargo update

# Report everything behind upstream: crates in all three cargo workspaces
# (majors included), bun and the GUI's JS, the MLX pin, the Rust toolchain,
# the maturin floor, GitHub Actions, and duplicate crate versions. Read-only.
deps-check:
    scripts/deps-check.sh

alias outdated := deps-check

# One version per dependency: members inherit every dependency from the root
# [workspace.dependencies], and the GUI and fuzz workspaces repeat the root's
# requirement for every crate they share with it (CI: deny job).
check-workspace-deps:
    scripts/check-workspace-deps.sh

# ─── Utilities ──────────────────────────────────────────────────────

# Clean build artifacts
clean:
    cargo clean

# Build documentation
doc:
    cargo doc --workspace --exclude pmetal-py --no-deps --open

# Show dependency tree
deps:
    cargo tree --workspace-only

# License, advisory, ban and source policy from deny.toml (CI: deny job).
# The GUI is a separate workspace, so it gets its own run against the same file.
deny:
    cargo deny check
    cargo deny --manifest-path crates/pmetal-gui/src-tauri/Cargo.toml check

# Security advisories only, with deny.toml's reviewed ignores applied
audit:
    cargo deny check advisories
    cargo deny --manifest-path crates/pmetal-gui/src-tauri/Cargo.toml check advisories

# ─── Benchmarks ─────────────────────────────────────────────────────

# Run all pmetal-metal benchmarks (ANE kernels + kernel dispatch)
bench-metal:
    cargo bench -p pmetal-metal

# Run the Metal kernel-dispatch benchmark only. Establishes a baseline
# for the `dispatch_simple_kernel` helper; re-run after shader/dispatch
# changes to catch regressions.
bench-metal-dispatch:
    cargo bench -p pmetal-metal --bench kernel_dispatch

# ─── Fuzzing ────────────────────────────────────────────────────────

# Run GGUF fuzz tests (CI: fuzz job, 5 min)
fuzz duration="300":
    cargo +nightly fuzz run gguf_reader -- -max_total_time={{ duration }}

# ─── Kani Verification ──────────────────────────────────────────────

# Install Kani verifier
kani-setup:
    cargo install --locked kani-verifier
    cargo kani setup

# Run Kani verification on distributed primitives
kani-verify:
    cargo kani --package pmetal-distributed

# ─── Training Shortcuts ─────────────────────────────────────────────

# Run a training job
train model_path output_dir:
    cargo run --release --bin pmetal -- train --model {{ model_path }} --output {{ output_dir }}

# Launch TUI dashboard
tui:
    cargo run --release --bin pmetal -- tui
