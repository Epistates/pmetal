#!/usr/bin/env bash
#
# Publish all pmetal crates to crates.io in topological order.
#
# Prerequisites:
#   - `cargo login` has been run with a valid API token
#   - All workspace Cargo.toml git dependencies have been replaced with
#     crates.io version references (e.g., mlx-rs = "0.25.x")
#   - The workspace version in Cargo.toml matches the intended release
#
# Usage:
#   ./scripts/publish.sh            # Publish all crates
#   ./scripts/publish.sh --dry-run  # Verify without publishing
#
set -euo pipefail

DRY_RUN=false
DELAY=600  # seconds between publishes (crates.io rate limit)

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    DELAY=0
    echo "=== DRY RUN MODE ==="
fi

# Topological publish order (respects internal dependency graph)
CRATES=(
    pmetal-core          #  1. no internal deps
    pmetal-distributed   #  2. no internal deps
    pmetal-gguf          #  3. depends on: core
    pmetal-metal         #  4. depends on: core
    pmetal-hub           #  5. depends on: core
    pmetal-data          #  6. depends on: core
    pmetal-mlx           #  7. depends on: core, metal
    pmetal-mhc           #  8. depends on: core, metal (optional)
    pmetal-models        #  9. depends on: core, gguf, metal, mlx
    pmetal-vocoder       # 10. depends on: core, mlx
    pmetal-merge         # 11. depends on: core, mlx
    pmetal-distill       # 12. depends on: core, mlx, metal (optional)
    pmetal-lora          # 13. depends on: core, gguf, metal, mlx, models
    pmetal-trainer       # 14. depends on: core, data, distill, lora, metal, mlx, models
    pmetal-cli           # 15. depends on: all
)

TOTAL=${#CRATES[@]}
PUBLISHED=0
FAILED=()

for crate in "${CRATES[@]}"; do
    PUBLISHED=$((PUBLISHED + 1))
    echo ""
    echo "[$PUBLISHED/$TOTAL] Publishing $crate..."

    if $DRY_RUN; then
        cargo publish -p "$crate" --dry-run --allow-dirty 2>&1
    else
        if cargo publish -p "$crate" 2>&1; then
            echo "  ✓ $crate published"
        else
            echo "  ✗ $crate FAILED"
            FAILED+=("$crate")
            echo "Stopping — fix the failure and re-run from $crate"
            break
        fi

        # Wait for crates.io index to propagate (skip after last crate)
        if [[ $PUBLISHED -lt $TOTAL ]]; then
            echo "  Waiting ${DELAY}s for crates.io index propagation..."
            sleep "$DELAY"
        fi
    fi
done

echo ""
echo "=== Summary ==="
echo "Published: $PUBLISHED / $TOTAL"

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "Failed: ${FAILED[*]}"
    exit 1
else
    echo "All crates published successfully."
fi
