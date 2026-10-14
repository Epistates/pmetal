//! Compile-fail snapshots for `#[derive(JobSpec)]`: each rejected input must
//! fail with a message that names the problem, at the span of the key at fault.
//!
//! Regenerate the `.stderr` snapshots with `TRYBUILD=overwrite` after an
//! intentional change to a message, and review the diff.

#[test]
fn job_spec_derive_rejects_bad_input() {
    trybuild::TestCases::new().compile_fail("tests/ui/*.rs");
}
