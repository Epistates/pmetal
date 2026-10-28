#!/usr/bin/env bash
#
# Publish all pmetal crates to crates.io in topological order.
#
# Prerequisites:
#   - `cargo login` has been run with a valid API token
#   - All workspace Cargo.toml git dependencies have been replaced with
#     crates.io version references for the internal pmetal crates
#   - The workspace version in Cargo.toml matches the intended release
#
# Rate limits (new crates):
#   1 per 10 minutes, burst of 5. First 5 publish immediately,
#   then 10 min between each subsequent crate.
#
# Rate limits (new versions of existing crates):
#   1 per minute, burst of 30. Essentially no wait needed.
#
# Usage:
#   ./scripts/publish.sh            # Publish all crates (first-time)
#   ./scripts/publish.sh --dry-run  # Verify without publishing
#
set -euo pipefail

# Preflight checks
command -v cargo >/dev/null 2>&1 || { echo "cargo not found in PATH" >&2; exit 1; }
[[ -f "Cargo.toml" ]] || { echo "Must be run from workspace root" >&2; exit 1; }

DRY_RUN=false
# For first-time crate creation: burst of 5, then 10 min between each.
# For version bumps of existing crates: set to 60 (or even 0 with burst of 30).
NEW_CRATE_DELAY=600
BURST_SIZE=5

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

if $DRY_RUN; then
    echo "=== DRY RUN MODE ==="
fi

# The order is derived from the manifests (scripts/publish-order.sh), so it
# follows the workspace instead of a list that can go stale. Check it first:
# a crate missing from it, or one published before a crate it needs, would
# only show up halfway through a release. So would a crate crates.io refuses,
# which a dry run never asks it about.
scripts/check-publish-order.sh
scripts/check-crates-io.sh

CRATES=()
while IFS= read -r crate; do
    CRATES+=("$crate")
done < <(scripts/publish-order.sh)

TOTAL=${#CRATES[@]}
SUCCEEDED=0
ATTEMPTED=0
FAILED=()

for crate in "${CRATES[@]}"; do
    ATTEMPTED=$((ATTEMPTED + 1))
    echo ""
    echo "[$ATTEMPTED/$TOTAL] Publishing $crate..."

    if $DRY_RUN; then
        cargo publish -p "$crate" --dry-run --allow-dirty 2>&1
    else
        if cargo publish -p "$crate" 2>&1; then
            SUCCEEDED=$((SUCCEEDED + 1))
            echo "  ✓ $crate published"
        else
            echo "  ✗ $crate FAILED" >&2
            FAILED+=("$crate")
            echo "Stopping — fix the failure and re-run from $crate" >&2
            break
        fi

        # Rate limit: burst of 5 new crates, then wait between each
        if [[ $ATTEMPTED -lt $TOTAL ]]; then
            if [[ $ATTEMPTED -ge $BURST_SIZE ]]; then
                echo "  Waiting ${NEW_CRATE_DELAY}s for crates.io rate limit..."
                sleep "$NEW_CRATE_DELAY"
            else
                # Within burst window, still need brief wait for index propagation
                echo "  Waiting 30s for index propagation..."
                sleep 30
            fi
        fi
    fi
done

echo ""
echo "=== Summary ==="
echo "Succeeded: $SUCCEEDED / $TOTAL"

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "Failed: ${FAILED[*]}" >&2
    exit 1
else
    echo "All crates published successfully."
fi
