#!/usr/bin/env bash
# The crates to publish, one per line, dependencies first.
#
# Derived from `cargo metadata`, never written down, so it cannot go stale
# when a crate is added, renamed or removed. A crate is published when its
# manifest does not say `publish = false`. A crate comes after every
# workspace crate it needs from the registry: its normal and build
# dependencies, and the dev-dependencies that name a version (a path-only
# one is stripped when publishing). Ties go alphabetically, so the order is
# the same on every machine.
#
# `scripts/check-publish-order.sh` checks the result independently.
set -euo pipefail
cd "$(dirname "$0")/.."

cargo metadata --no-deps --format-version 1 --locked | jq -r '
    [.packages[] | select(.publish != [])] as $crates
    | ($crates | map(.name)) as $names
    | ($crates
        | map({
            key: .name,
            value: ([.dependencies[]
                | select(.path != null)
                | select(.name as $d | $names | index($d))
                | select(.kind != "dev" or .req != "*")
                | .name] | unique)
          })
        | from_entries) as $raw
    # A crate never waits for itself (a dev-dependency used to turn on its own features).
    | ($raw | with_entries(.key as $k | .value |= map(select(. != $k)))) as $deps
    | reduce range(0; $names | length) as $_ ({done: [], left: ($names | sort)};
        (.done as $done
         | [.left[] | select(. as $n | ($deps[$n] - $done | length) == 0)]) as $ready
        | if ($ready | length) == 0
          then error("dependency cycle among: \(.left | join(", "))")
          else .done += [$ready[0]] | .left -= [$ready[0]]
          end)
    | .done[]'
