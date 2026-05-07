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
preflight: fmt-check lint lint-all-features test test-release check-gui check-version check-lockfile
    @echo ""
    @echo "All preflight checks passed -- safe to publish {{ version }}"

# ─── Formatting ─────────────────────────────────────────────────────

# Format code
fmt:
    cargo fmt --all

# Check formatting without changing files (CI: check job)
fmt-check:
    cargo fmt --all -- --check

# ─── Linting ────────────────────────────────────────────────────────

# Clippy with default features, excluding pmetal-py (CI: check job)
lint:
    cargo clippy --workspace --all-targets --exclude pmetal-py -- -D warnings

# Clippy with ALL features enabled (catches cfg-gated field mismatches)
lint-all-features:
    cargo clippy --workspace --all-targets --all-features -- -D warnings

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

# Build CLI binary (mirrors release CI)
build-cli:
    cargo build --release -p pmetal

# Check GUI compiles (mirrors release CI -- catches cfg mismatches)
check-gui:
    cargo check --manifest-path crates/pmetal-gui/src-tauri/Cargo.toml

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

# Verify Cargo.lock is up-to-date and no yanked crates
check-lockfile:
    cargo update --locked 2>&1 || (echo "FAIL: Cargo.lock is out of date -- run cargo update" && exit 1)
    @echo "Cargo.lock is current, no yanked crates"

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

# Show deps behind latest (including semver-breaking)
outdated:
    cargo update --verbose 2>&1 | grep -E "Unchanged|Locking" || echo "All deps are latest"

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

# Run security audit
audit:
    cargo audit

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
