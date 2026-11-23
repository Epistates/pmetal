#!/usr/bin/env bash
#
# Publish all pmetal crates to crates.io in dependency order.
#
# Prerequisites:
#   - `cargo login` has been run with a valid API token
#   - The workspace version in Cargo.toml matches the intended release
#
# Running it again after a failure is safe: a crate whose version is already
# on crates.io is skipped, so the release resumes where it stopped.
#
# Rate limits: crates.io takes 5 new crates at once and then one every 10
# minutes, and 30 new versions of existing crates at once and then one a
# minute. When it refuses an upload for that reason it says when to try again,
# and the script waits until then and retries. `cargo publish` itself waits
# until each crate is in the index, so the next crate can depend on it.
#
# Usage:
#   ./scripts/publish.sh            # Publish every crate
#   ./scripts/publish.sh --dry-run  # Package and build every crate, upload nothing
#
set -euo pipefail

# Preflight checks
command -v cargo >/dev/null 2>&1 || { echo "cargo not found in PATH" >&2; exit 1; }
command -v jq >/dev/null 2>&1 || { echo "jq not found in PATH" >&2; exit 1; }
[[ -f "Cargo.toml" ]] || { echo "Must be run from workspace root" >&2; exit 1; }

DRY_RUN=false
USER_AGENT="pmetal-publish (https://github.com/Epistates/pmetal)"
# crates.io's longest wait is ten minutes, for a brand-new crate, so a crate
# that is still refused after three waits, or a wait of over half an hour,
# means something else is wrong.
MAX_RATE_LIMIT_RETRIES=3
MAX_RATE_LIMIT_WAIT=1800

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

METADATA=$(cargo metadata --no-deps --format-version 1 --locked)

version_of() {
    jq -r --arg name "$1" '.packages[] | select(.name == $name) | .version' <<<"$METADATA"
}

# Whether crates.io already has this version of the crate, read from the
# sparse index that cargo itself uses. Returns 0 if it has, 1 if it hasn't,
# and 2 if the index could not be read.
is_published() {
    local name path body status
    name=$(tr '[:upper:]' '[:lower:]' <<<"$1")
    case ${#name} in
        1) path="1/$name" ;;
        2) path="2/$name" ;;
        3) path="3/${name:0:1}/$name" ;;
        *) path="${name:0:2}/${name:2:2}/$name" ;;
    esac
    body=$(mktemp)
    if ! status=$(curl -sS -A "$USER_AGENT" -o "$body" -w '%{http_code}' \
        "https://index.crates.io/$path"); then
        rm -f "$body"
        return 2
    fi
    case $status in
        200)
            if jq -e --arg v "$2" 'select(.vers == $v)' "$body" >/dev/null; then
                rm -f "$body"
                return 0
            fi
            ;;
        404) ;;
        *)
            echo "  The crates.io index answered HTTP $status for $1" >&2
            rm -f "$body"
            return 2
            ;;
    esac
    rm -f "$body"
    return 1
}

# Sleep until an HTTP date such as "Thu, 23 Oct 2026 05:00:00 GMT", plus a
# few seconds of margin. GNU date reads it with -d, BSD date with -j -f.
wait_until() {
    local target now
    if ! target=$(LC_ALL=C date -u -d "$1" +%s 2>/dev/null ||
        LC_ALL=C date -u -j -f '%a, %d %b %Y %T GMT' "$1" +%s 2>/dev/null); then
        echo "  Could not read the time crates.io gave: $1" >&2
        return 1
    fi
    now=$(date -u +%s)
    if ((target - now > MAX_RATE_LIMIT_WAIT)); then
        echo "  crates.io asks to wait until $1, more than ${MAX_RATE_LIMIT_WAIT}s; giving up." >&2
        return 1
    fi
    if ((target > now)); then
        echo "  crates.io asks to wait until $1; waiting $((target - now + 5))s..."
        sleep $((target - now + 5))
    fi
}

# Publish one crate, waiting out crates.io's rate limit as often as it asks.
publish() {
    local log retry_after attempt=0
    log=$(mktemp)
    while true; do
        if cargo publish -p "$1" 2>&1 | tee "$log"; then
            rm -f "$log"
            return 0
        fi
        retry_after=$(sed -n 's/.*Please try again after \([^.]* GMT\).*/\1/p' "$log" | head -n 1)
        rm -f "$log"
        if [[ -z $retry_after ]]; then
            return 1
        fi
        attempt=$((attempt + 1))
        if ((attempt > MAX_RATE_LIMIT_RETRIES)); then
            echo "  crates.io refused $1 under its rate limit $attempt times; giving up." >&2
            return 1
        fi
        wait_until "$retry_after" || return 1
        log=$(mktemp)
    done
}

CRATES=()
while IFS= read -r crate; do
    CRATES+=("$crate")
done < <(scripts/publish-order.sh)

TOTAL=${#CRATES[@]}
PUBLISHED=0
SKIPPED=0
ATTEMPTED=0
FAILED=()

for crate in "${CRATES[@]}"; do
    ATTEMPTED=$((ATTEMPTED + 1))
    version=$(version_of "$crate")
    echo ""
    echo "[$ATTEMPTED/$TOTAL] $crate $version"

    if $DRY_RUN; then
        cargo publish -p "$crate" --dry-run --allow-dirty 2>&1
        continue
    fi

    on_index=0
    is_published "$crate" "$version" || on_index=$?
    if [[ $on_index -eq 0 ]]; then
        echo "  already on crates.io, skipped"
        SKIPPED=$((SKIPPED + 1))
        continue
    elif [[ $on_index -eq 2 ]]; then
        FAILED+=("$crate")
        echo "Stopping: could not read the crates.io index. Run the script again to resume." >&2
        break
    fi

    if publish "$crate"; then
        PUBLISHED=$((PUBLISHED + 1))
        echo "  ✓ $crate published"
    else
        echo "  ✗ $crate FAILED" >&2
        FAILED+=("$crate")
        echo "Stopping: fix the failure and run the script again. Crates already on crates.io are skipped." >&2
        break
    fi
done

if $DRY_RUN; then
    exit 0
fi

echo ""
echo "=== Summary ==="
echo "Published: $PUBLISHED, already on crates.io: $SKIPPED, of $TOTAL"

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "Failed: ${FAILED[*]}" >&2
    exit 1
else
    echo "All crates are on crates.io."
fi
