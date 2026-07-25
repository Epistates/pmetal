//! Shared scaffolding for this crate's parity tests.
//!
//! Shard loading and keyed lookup come from [`pmetal_mlx::test_utils`], which is
//! where the crate-agnostic parity helpers live. Only `fixture_path` is local:
//! it keys off this crate's `CARGO_MANIFEST_DIR`, so each crate needs its own.

use std::path::PathBuf;

pub use pmetal_mlx::test_utils::{load_shard, ref_tensor};

/// Resolve `tests/fixtures/<name>` relative to this crate.
pub fn fixture_path(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push(name);
    p
}
