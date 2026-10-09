#!/usr/bin/env bash
# Report everything versioned in the repo that is behind upstream. Read-only:
# it changes no file, and exits 0 whatever it finds.
#
#   cargo     requirements behind the latest release (majors included) and
#             lockfile entries behind a compatible release, per workspace
#   bun       the GUI's JS dependencies and the pinned bun itself
#   MLX       the bridge's pin against the latest release and upstream main
#   Rust      rust-toolchain.toml against the latest stable
#   maturin   the pyproject floor against the latest release
#   actions   each `uses:` in .github/workflows against the action's newest tag
#   dups      crates that resolve to more than one version, per lockfile
#
# Needs network access, cargo-edit (`cargo upgrade`), bun, jq and curl.
set -uo pipefail
cd "$(dirname "$0")/.."

CARGO_DIRS=(. crates/pmetal-gui/src-tauri fuzz)

section() { printf '\n== %s\n' "$*"; }

section "cargo: requirements behind the latest release"
if cargo upgrade --version >/dev/null 2>&1; then
    for dir in "${CARGO_DIRS[@]}"; do
        echo "-- $dir"
        # --ignore-rust-version: without it, cargo upgrade proposes the newest
        # version whose declared rust-version fits the MSRV, which can be a
        # downgrade.
        cargo upgrade --manifest-path "$dir/Cargo.toml" --dry-run --incompatible \
            --ignore-rust-version --pinned allow 2>&1 |
            grep -vE '^ *(Checking|Updating)|aborting upgrade|Re-run with|^ *(latest|local):' || true
    done
else
    echo "cargo-edit is not installed: cargo install cargo-edit"
fi

section "cargo: lockfile entries behind a compatible release"
for dir in "${CARGO_DIRS[@]}"; do
    out=$(cargo update --manifest-path "$dir/Cargo.toml" --dry-run --verbose 2>&1 |
        grep -E '^ *(Updating|Unchanged) [a-z]' | grep -v 'crates.io index' || true)
    updatable=$(grep -c 'Updating' <<<"$out" || true)
    held=$(grep 'Unchanged' <<<"$out" || true)
    echo "-- $dir: $updatable updatable by \`cargo update\`"
    if [[ -n $held ]]; then
        echo "   held back by an upstream requirement (cargo update -p NAME --precise VERSION --dry-run names it):"
        sed 's/^ */   /' <<<"$held"
    fi
done

section "bun"
(
    cd crates/pmetal-gui || exit
    pinned=$(jq -r '.packageManager' package.json)
    latest=$(curl -fsSL https://registry.npmjs.org/bun/latest | jq -r .version)
    echo "packageManager: $pinned, latest bun: $latest"
    bun outdated 2>&1 || true
)

section "MLX"
pin=$(sed -nE 's/^const BUNDLED_MLX_GIT_TAG: &str = "([^"]+)";/\1/p' crates/pmetal-bridge/build.rs)
refs=$(git ls-remote https://github.com/ml-explore/mlx.git refs/heads/main 'refs/tags/v*')
main_sha=$(awk '$2 == "refs/heads/main" { print $1 }' <<<"$refs")
latest_tag=$(awk '$2 !~ /\^\{\}$/ && $2 ~ /^refs\/tags\/v[0-9.]+$/ { sub("refs/tags/", "", $2); print $2 }' <<<"$refs" |
    sort -V | tail -1)
latest_tag_sha=$(awk -v t="refs/tags/$latest_tag^{}" -v l="refs/tags/$latest_tag" \
    '$2 == t { peeled = $1 } $2 == l { plain = $1 } END { print peeled ? peeled : plain }' <<<"$refs")
echo "pinned:         $pin"
echo "latest release: $latest_tag ($latest_tag_sha)"
echo "main:           $main_sha"
if [[ $pin == "$main_sha" || $pin == "$latest_tag" ]]; then
    echo "up to date"
else
    echo "behind: compare https://github.com/ml-explore/mlx/compare/$pin...main"
fi

section "Rust toolchain"
pinned=$(sed -nE 's/^channel = "([^"]+)"/\1/p' rust-toolchain.toml)
stable=$(curl -fsSL https://static.rust-lang.org/dist/channel-rust-stable.toml |
    awk '/^\[pkg.rustc\]/ { f = 1 } f && /^version/ { print $3; exit }' | tr -d '"')
echo "rust-toolchain.toml: $pinned, latest stable: $stable"
msrv=$(sed -nE 's/^rust-version = "([^"]+)"/\1/p' Cargo.toml)
echo "workspace rust-version (MSRV): $msrv"

section "maturin"
floor=$(sed -nE 's/.*"maturin>=([0-9.]+),.*/\1/p' crates/pmetal-py/pyproject.toml)
latest=$(curl -fsSL https://pypi.org/pypi/maturin/json | jq -r .info.version)
echo "pyproject floor: $floor, latest: $latest"

section "GitHub Actions"
grep -rhoE 'uses: *[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+' .github/workflows |
    sed 's/uses: *//' | sort -u | while IFS=@ read -r action ref; do
    newest=$(git ls-remote --tags "https://github.com/$action.git" 2>/dev/null |
        awk '{ sub("refs/tags/", "", $2); sub(/\^\{\}$/, "", $2); print $2 }' |
        grep -E '^v[0-9]+$' | sort -V | tail -1)
    if [[ -z $newest ]]; then
        newest=$(git ls-remote --tags "https://github.com/$action.git" 2>/dev/null |
            awk '{ sub("refs/tags/", "", $2); sub(/\^\{\}$/, "", $2); print $2 }' |
            grep -E '^v?[0-9]+(\.[0-9]+)*$' | sort -V | tail -1)
    fi
    flag=""
    [[ -n $newest && $ref != "$newest" && $ref =~ ^v[0-9]+$ ]] && flag="  <- behind"
    printf '%-40s %-10s newest major: %s%s\n' "$action" "$ref" "${newest:-?}" "$flag"
done

section "duplicate crate versions (distinct versions per crate name)"
for dir in "${CARGO_DIRS[@]}"; do
    awk '/^name = /{ n = $3 } /^version = /{ print n, $3 }' "$dir/Cargo.lock" | sort -u |
        awk -v d="$dir" '{ c[$1]++ } END {
            for (k in c) if (c[k] > 1) { crates++; extra += c[k] - 1 }
            printf "-- %s: %d crates with more than one version (%d extra versions)\n", d, crates, extra
        }'
done
echo "   list them with: cargo tree --duplicates [--manifest-path DIR/Cargo.toml]"
