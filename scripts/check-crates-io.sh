#!/usr/bin/env bash
# Every published crate is one crates.io accepts, checked before any crate is
# uploaded.
#
#   scripts/check-crates-io.sh [METADATA_JSON]
#
# `cargo publish --dry-run` packages and builds each crate but never asks the
# registry, so a crate it passes can still be refused partway through a
# release, after the crates before it are published. METADATA_JSON is the
# output of `cargo metadata --no-deps --format-version 1`; by default it is
# read from the workspace. For each crate without `publish = false` it fails
# on
#   - no description, or no license (crates.io requires both);
#   - more than 5 keywords, or a keyword over 20 characters, or one that does
#     not start with a letter or digit and continue with letters, digits,
#     `_`, `-` or `+`;
#   - more than 5 categories, or one that is not a crates.io category slug
#     (crates.io drops an unknown one with only a warning);
#   - a package over 10 MiB, the most crates.io takes (it closes the upload
#     without saying why).
#
# The category slugs come from crates.io, so this needs network access.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ $# -gt 0 ]]; then
    metadata=$(cat "$1")
else
    metadata=$(cargo metadata --no-deps --format-version 1 --locked)
fi

slugs=$(curl -fsS -A "pmetal-preflight (https://github.com/Epistates/pmetal)" \
    https://crates.io/api/v1/category_slugs | jq -c '[.category_slugs[].slug]')

problems=$(jq -r --argjson slugs "$slugs" '
    .packages[] | select(.publish != []) | .name as $crate
    | (select((.description // "") == "") | "\($crate): no description"),
      (select((.license // "") == "" and (.license_file // "") == "")
          | "\($crate): no license"),
      (select((.keywords | length) > 5)
          | "\($crate): \(.keywords | length) keywords, crates.io takes at most 5"),
      (.keywords[] | select(length > 20 or (test("^[A-Za-z0-9][A-Za-z0-9_+-]*$") | not))
          | "\($crate): keyword \"\(.)\" (\(length) characters): at most 20, of letters, digits, _, - and +"),
      (select((.categories | length) > 5)
          | "\($crate): \(.categories | length) categories, crates.io takes at most 5"),
      (.categories[] | select(. as $c | $slugs | index($c) | not)
          | "\($crate): category \"\(.)\" is not a crates.io category")
' <<<"$metadata")

# The packages as `cargo publish` would upload them, unbuilt.
packages=()
while IFS= read -r name; do
    packages+=(-p "$name")
done < <(jq -r '.packages[] | select(.publish != []) | .name' <<<"$metadata")
if ! log=$(cargo package "${packages[@]}" --no-verify --allow-dirty 2>&1); then
    echo "$log"
    echo "FAIL: cargo package failed"
    exit 1
fi
target_dir=$(jq -r '.target_directory' <<<"$metadata")
limit=$((10 * 1024 * 1024))
while IFS=$'\t' read -r name version; do
    file="$target_dir/package/$name-$version.crate"
    size=$(wc -c <"$file")
    if ((size > limit)); then
        problems+=${problems:+$'\n'}"$name: the package is $((size / 1024 / 1024)) MiB, crates.io takes at most 10 (\`cargo package --list -p $name\` shows what is in it)"
    fi
done < <(jq -r '.packages[] | select(.publish != []) | [.name, .version] | @tsv' <<<"$metadata")

if [[ -n $problems ]]; then
    while IFS= read -r problem; do
        echo "FAIL: $problem"
    done <<<"$problems"
    exit 1
fi
echo "Every published crate is one crates.io accepts: description, license, keywords, categories and size"
