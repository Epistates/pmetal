//! Shared scaffolding for the architecture parity tests.
//!
//! Every `*_parity.rs` integration test loads a committed safetensors fixture
//! (reference activations + the raw weights that produced them) and diffs the
//! Rust forward pass against it. These helpers are the common plumbing —
//! shard loading, keyed tensor lookup, and fixture-path resolution — so each
//! test file carries only its architecture-specific config, walk, and
//! tolerances.
//!
//! The numerical primitives (`ParityReport`, `Tolerance`, `print_report_table`)
//! live in [`pmetal_mlx::test_utils`]; they are crate-agnostic. Fixture-path
//! resolution, by contrast, is intentionally here: it keys off this crate's
//! `CARGO_MANIFEST_DIR`, so it must live in the consuming crate's test tree.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use pmetal_bridge::compat::Array;

/// Load a safetensors shard into a name→tensor map.
pub fn load_shard(path: &Path) -> HashMap<String, Array> {
    let path_str = path.to_str().expect("utf8 path");
    let pairs = pmetal_bridge::inline_array::load_safetensors_shard(path_str)
        .unwrap_or_else(|| panic!("failed to load safetensors shard at {path_str:?}"));
    pairs.into_iter().collect()
}

/// Fetch a reference tensor by key, panicking with the missing key on absence.
pub fn ref_tensor<'a>(shard: &'a HashMap<String, Array>, key: &str) -> &'a Array {
    shard
        .get(key)
        .unwrap_or_else(|| panic!("reference shard missing key {key:?}"))
}

/// Resolve `tests/fixtures/<name>` relative to this crate.
pub fn fixture_path(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push(name);
    p
}
