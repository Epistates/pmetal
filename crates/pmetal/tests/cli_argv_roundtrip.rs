//! CLI argv round-trip integration tests.
//!
//! # Why tests live in two places
//!
//! The `Cli` struct is defined in `crates/pmetal/src/main.rs` (a binary crate)
//! and is therefore not accessible from integration tests in this `tests/`
//! directory. The executable round-trip tests (one per spec) are in the
//! `argv_roundtrip` module at the bottom of `src/main.rs` so they can see
//! the private `Cli` type.
//!
//! This file serves two purposes:
//!   1. A reference for how to write *process-level* round-trip tests once the
//!      CLI types are migrated to `lib.rs` (see `src/cli/README.md`).
//!   2. A compile-time gate: if this file doesn't compile, the test run fails.
//!
//! # Running the unit-level tests (the actual spec validators)
//!
//! ```sh
//! cargo test -p pmetal --features trainer -- argv_roundtrip
//! ```
//!
//! # Running a specific spec
//!
//! ```sh
//! cargo test -p pmetal --features trainer -- argv_roundtrip::train_spec_round_trip
//! ```
//!
//! # How to extend to a new spec
//!
//! 1. Add the spec to `pmetal_core::jobs` (already done in Phase 3).
//! 2. Add an `#[test]` in `src/main.rs::argv_roundtrip` following the pattern:
//!
//!    ```rust,ignore
//!    #[test]
//!    fn myspec_spec_round_trip() {
//!        let mut spec = MySpec::default();
//!        // Set all required fields.
//!        spec.required_field = "value".into();
//!        let result = try_parse("my-subcommand", spec.to_argv());
//!        assert!(result.is_ok(), "MySpec argv failed: {}", result.unwrap_err());
//!    }
//!    ```
//!
//! 3. If the test fails, the spec's `argv = "..."` attribute does NOT match
//!    the CLI flag. Fix the spec — the CLI is authoritative.

// Nothing executable here — the real tests are in src/main.rs::argv_roundtrip.
// This file must compile cleanly for `cargo test -p pmetal --test cli_argv_roundtrip`
// to pass.
