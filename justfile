# PMetal Development Justfile
# Run `just --list` to see all available recipes

# Default recipe - show help
default:
    @just --list

# Build the project
build:
    cargo build --release

# Build in debug mode
build-debug:
    cargo build

# Check compilation without building
check:
    cargo check

# Run all tests (single-threaded for Metal GPU compatibility)
test:
    cargo test -- --test-threads=1

# Run tests for a specific package
test-pkg pkg:
    cargo test --package {{pkg}} -- --test-threads=1

# Run tests with output shown
test-verbose:
    cargo test -- --test-threads=1 --nocapture

# Run only unit tests (no doc tests)
test-unit:
    cargo test --lib -- --test-threads=1

# Run clippy lints
lint:
    cargo clippy --all-targets --all-features -- -D warnings

# Format code
fmt:
    cargo fmt

# Check formatting without changing files
fmt-check:
    cargo fmt -- --check

# Run all CI checks (fmt, clippy, test)
ci: fmt-check lint test

# Clean build artifacts
clean:
    cargo clean

# Build documentation
doc:
    cargo doc --no-deps --open

# Run a specific trainer example
train model_path output_dir:
    cargo run --release --bin pmetal-cli -- train --model {{model_path}} --output {{output_dir}}

# Run benchmarks (if any)
bench:
    cargo bench

# Update dependencies
update:
    cargo update

# Show dependency tree
deps:
    cargo tree

# Check for outdated dependencies
outdated:
    cargo outdated -R

# Run security audit
audit:
    cargo audit

# Profile memory usage during tests
test-mem:
    cargo test -- --test-threads=1 2>&1 | grep -E "(memory|MB|GB)"

# Watch for changes and run tests
watch:
    cargo watch -x 'test -- --test-threads=1'

# Run specific test by name
test-name name:
    cargo test {{name}} -- --test-threads=1 --nocapture
