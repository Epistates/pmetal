//! Shared scaffolding for the architecture parity tests.
//!
//! Every `*_parity.rs` integration test loads a committed safetensors fixture
//! (reference activations + the raw weights that produced them) and diffs the
//! Rust forward pass against it. Shard loading and keyed tensor lookup are
//! crate-agnostic, so they live in [`pmetal_mlx::test_utils`] alongside the
//! numerical primitives (`ParityReport`, `Tolerance`, `print_report_table`) and
//! are re-exported here for the test files' `use common::{…}` imports.
//!
//! Fixture-path resolution, by contrast, is intentionally local: it keys off
//! this crate's `CARGO_MANIFEST_DIR`, so it must live in the consuming crate's
//! test tree.

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
