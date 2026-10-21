#!/usr/bin/env bash
# The publish order is complete and in dependency order, checked from the
# manifests rather than from the script that derives it.
#
#   scripts/check-publish-order.sh [ORDER_FILE]
#
# ORDER_FILE holds one crate per line; by default it is the output of
# `scripts/publish-order.sh`. It fails when
#   - a publishable workspace crate (no `publish = false`) is missing, or
#     listed twice, or a listed name is not a publishable workspace crate;
#   - a crate is listed before a workspace crate it needs from the registry;
#   - a publishable crate needs a workspace crate that is `publish = false`;
#   - a dependency on a workspace crate names no version (the registry
#     rejects a path-only dependency).
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ $# -gt 0 ]]; then
    order=$(cat "$1")
else
    order=$(scripts/publish-order.sh)
fi

metadata=$(cargo metadata --no-deps --format-version 1 --locked)

errors=0
fail() {
    echo "FAIL: $*"
    errors=$((errors + 1))
}

publishable=$(jq -r '.packages[] | select(.publish != []) | .name' <<<"$metadata" | sort)
unpublished=$(jq -r '.packages[] | select(.publish == []) | .name' <<<"$metadata" | sort)

# Set equality, and no repeats.
listed=$(sort <<<"$order")
while read -r name; do
    [[ -n $name ]] || continue
    grep -qx -- "$name" <<<"$listed" || fail "$name is publishable but missing from the publish order"
done <<<"$publishable"
while read -r name; do
    [[ -n $name ]] || continue
    grep -qx -- "$name" <<<"$publishable" || fail "$name is in the publish order but is not a publishable workspace crate"
done <<<"$listed"
duplicates=$(uniq -d <<<"$listed")
[[ -z $duplicates ]] || fail "listed more than once: $(tr '\n' ' ' <<<"$duplicates")"

# Every edge a publish needs goes forward.
position() {
    { grep -nx -- "$1" <<<"$order" || true; } | head -1 | cut -d: -f1
}
edges=$(jq -r '
    .packages[] | select(.publish != []) | .name as $p
    | .dependencies[]
    | select(.path != null and .name != $p)
    | select(.kind != "dev" or .req != "*")
    | "\($p) \(.name) \(.kind // "normal") \(.req)"' <<<"$metadata")
while read -r crate dep kind req; do
    [[ -n $crate ]] || continue
    if grep -qx -- "$dep" <<<"$unpublished"; then
        fail "$crate needs $dep ($kind), which is publish = false"
        continue
    fi
    if [[ $req == "*" ]]; then
        fail "$crate's $kind dependency on $dep names no version"
    fi
    here=$(position "$crate")
    there=$(position "$dep")
    if [[ -n $here && -n $there && $there -gt $here ]]; then
        fail "$crate (#$here) is published before $dep (#$there), which it needs"
    fi
done <<<"$edges"

if [[ $errors -gt 0 ]]; then
    echo "$errors problem(s) with the publish order"
    exit 1
fi
echo "publish order OK ($(wc -l <<<"$order" | tr -d ' ') crates)"
