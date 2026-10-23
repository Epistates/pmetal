//! `checked` reports a failure anywhere in a chain of ops.

use pmetal_runtime::{
    InlineArray, RuntimeError, check_last_error, check_unobserved_error, checked,
};

fn vector(values: &[f32]) -> InlineArray {
    InlineArray::from_f32_slice(values, &[values.len() as i32])
}

/// The failing op returns a placeholder, the ops after it succeed on it, and
/// the last-op slot reads clean: `check_last_error` after the chain says Ok.
/// `checked` reads the first failure instead.
#[test]
fn an_error_in_the_middle_of_a_chain_is_reported() {
    let a = vector(&[1.0, 2.0]);
    let bad = vector(&[1.0, 2.0, 3.0]);

    // What the per-op check sees: nothing.
    let plain = pmetal_runtime::expect_runtime_errors(|| a.matmul(&bad).add(&a).sum_all());
    assert_eq!(check_last_error(), Ok(()), "the last op cleared its slot");
    // ...while the failure is still pending for whoever reads it.
    let pending = check_unobserved_error().expect_err("the matmul failure is still there");
    assert!(pending.to_string().contains("[matmul]"), "{pending}");
    drop(plain);

    let err = checked(|| a.matmul(&bad).add(&a).sum_all()).expect_err("a shape mismatch");
    match err {
        RuntimeError::CxxException(msg) => assert!(msg.contains("[matmul]"), "{msg}"),
        other => panic!("expected a C++ exception, got: {other}"),
    }
    // Reading it cleared it.
    assert_eq!(check_unobserved_error(), Ok(()));
}

#[test]
fn a_chain_that_succeeds_returns_its_value() {
    let a = vector(&[1.0, 2.0, 3.0]);
    let total = checked(|| a.multiply(&a).sum_all()).expect("no failure");
    assert_eq!(total.item_f32(), 14.0);
}

/// `eval` read the pending error into the token it returns, which clears it
/// from both channels; a caller that dropped the token (`a.eval();`, the usual
/// call) discarded the failure of whatever op came before, for good.
#[test]
fn eval_does_not_swallow_an_earlier_error() {
    let a = vector(&[1.0, 2.0]);
    let bad = vector(&[1.0, 2.0, 3.0]);
    let broken = pmetal_runtime::expect_runtime_errors(|| a.matmul(&bad));
    broken.eval(); // the token is dropped, as nearly every caller does

    let pending = check_unobserved_error();
    assert!(
        pending.is_err(),
        "the matmul failure was discarded by an eval whose token nobody read"
    );
}

/// The token still reports it for a caller that does read it.
#[test]
fn the_eval_token_reports_an_earlier_error() {
    let a = vector(&[1.0, 2.0]);
    let bad = vector(&[1.0, 2.0, 3.0]);
    let broken = pmetal_runtime::expect_runtime_errors(|| a.matmul(&bad));
    let err = broken.eval().into_result().expect_err("the matmul failed");
    assert!(err.to_string().contains("[matmul]"), "{err}");
    assert_eq!(check_unobserved_error(), Ok(()));
}

/// An error nobody read before the call is not lost, and not blamed on the
/// closure's own ops either: it is what comes back.
#[test]
fn an_earlier_unread_error_is_returned() {
    let a = vector(&[1.0, 2.0]);
    let bad = vector(&[1.0, 2.0, 3.0]);
    let _ = pmetal_runtime::expect_runtime_errors(|| a.matmul(&bad));

    let err = checked(|| a.add(&a)).expect_err("the earlier matmul failure");
    assert!(err.to_string().contains("[matmul]"), "{err}");
    assert_eq!(check_unobserved_error(), Ok(()));
}
