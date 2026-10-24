#!/usr/bin/env bash
# One version per dependency, enforced.
#
# 1. Every dependency of a root-workspace member comes from the root's
#    [workspace.dependencies]: `dep = { workspace = true, ... }` or
#    `dep.workspace = true`. A member that names a version, path or git source
#    itself fails.
# 2. Every [workspace.dependencies] entry is inherited by some member.
# 3. The GUI (crates/pmetal-gui/src-tauri) and fuzz workspaces can't inherit
#    from the root, so every crate they share with it must carry the same
#    requirement string as the root's entry.
#
# Manifests here keep one dependency per line, which is what this parses. A
# `[dependencies.foo]` table header is rejected outright for that reason.
set -euo pipefail
cd "$(dirname "$0")/.."

errors=0
fail() {
    echo "FAIL: $*"
    errors=$((errors + 1))
}

# Prints `section<TAB>name<TAB>rest-of-line` for every dependency line.
dep_lines() {
    awk '
        /^\[/ { sec = $0; next }
        sec ~ /dependencies\]$/ && /^[A-Za-z0-9_-]+(\.[A-Za-z0-9_-]+)? *=/ {
            name = $0; sub(/ *=.*/, "", name)
            rest = $0; sub(/^[^=]*= */, "", rest)
            printf "%s\t%s\t%s\n", sec, name, rest
        }
        /^\[([A-Za-z0-9_.-]+\.)?(dev-|build-)?dependencies\.[A-Za-z0-9_-]+\]$/ {
            printf "%s\t%s\t%s\n", $0, "<table>", ""
        }
    ' "$1"
}

# The version requirement of a dependency value: `"1.0"` or `{ version = "1.0", ... }`.
req_of() {
    sed -nE 's/^"([^"]*)".*/\1/p; s/.*version *= *"([^"]*)".*/\1/p' <<<"$1" | head -1
}

members=$(cargo metadata --no-deps --format-version 1 --locked |
    jq -r '.packages[].manifest_path')

for manifest in $members; do
    rel=${manifest#"$PWD/"}
    package=$(awk -F'"' '/^name *=/ { print $2; exit }' "$manifest")
    while IFS=$'\t' read -r sec name rest; do
        if [[ $sec == "[dev-dependencies]" && $name == "$package" && $rest =~ ^\{\ *path\ *=\ *\"\.\" ]]; then
            # A crate's tests turning on one of its own features. No version,
            # so publishing strips it rather than asking the registry for the
            # version being published.
            continue
        elif [[ $sec == "[dev-dependencies]" && $rest =~ ^\{\ *path\ *=\ *\"([^\"]+)\"\ *\}$ ]] &&
            grep -qx "$(cd "$(dirname "$manifest")/${BASH_REMATCH[1]}" 2>/dev/null && pwd)/Cargo.toml" <<<"$members"; then
            # A test-only edge to another workspace crate that is published
            # after this one (pmetal-runtime's tests holding its loader and
            # AdamW to pmetal-core's). No version, so publishing strips it and
            # the publish order does not have to account for it; a versioned
            # entry would need the other crate on the registry first.
            continue
        elif [[ $name == "<table>" ]]; then
            fail "$rel: $sec: write it as one line, \`dep = { workspace = true }\`"
        elif [[ $name == *.workspace && $rest == true* ]]; then
            continue
        elif [[ $name == *.* ]]; then
            fail "$rel: $sec $name: inherit it from [workspace.dependencies]"
        elif [[ ! $rest =~ workspace\ *=\ *true ]]; then
            fail "$rel: $sec $name = $rest: move the requirement into the root [workspace.dependencies] and use \`{ workspace = true }\`"
        elif [[ $rest =~ (^|[{,\ ])(version|path|git|branch|tag|rev|registry)\ *= ]]; then
            fail "$rel: $sec $name: an inherited dependency can't also name a source or version"
        fi
    done < <(dep_lines "$manifest")
done

# Root workspace requirements, `name<TAB>req`.
root_reqs=$(awk '
    /^\[/ { in_ws = ($0 == "[workspace.dependencies]"); next }
    in_ws && /^[A-Za-z0-9_-]+ *=/ {
        name = $0; sub(/ *=.*/, "", name)
        rest = $0; sub(/^[^=]*= */, "", rest)
        printf "%s\t%s\n", name, rest
    }
' Cargo.toml)

# An entry no member inherits is a version nobody gets reminded to bump.
while IFS=$'\t' read -r name _; do
    used=0
    for manifest in $members; do
        if grep -qE "^${name}( *=|\.workspace)" "$manifest"; then
            used=1
            break
        fi
    done
    ((used)) || fail "Cargo.toml: [workspace.dependencies] $name: no member uses it"
done <<<"$root_reqs"

for manifest in crates/pmetal-gui/src-tauri/Cargo.toml fuzz/Cargo.toml; do
    while IFS=$'\t' read -r sec name rest; do
        [[ $rest =~ path\ *= ]] && continue
        root_rest=$(awk -F'\t' -v n="$name" '$1 == n { print $2 }' <<<"$root_reqs")
        [[ -z $root_rest ]] && continue
        want=$(req_of "$root_rest")
        got=$(req_of "$rest")
        if [[ $want != "$got" ]]; then
            fail "$manifest: $name = \"$got\", the root workspace has \"$want\""
        fi
    done < <(dep_lines "$manifest")
done

if ((errors > 0)); then
    echo "FAIL: $errors dependency declaration(s) out of line with the root [workspace.dependencies]"
    exit 1
fi
echo "Every dependency comes from the root [workspace.dependencies]"
