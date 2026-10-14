//! `CompiledFn`: the Rust closure MLX traces and replays.

use pmetal_bridge::inline_array::{clear_cache, get_active_memory};
use pmetal_bridge::{CompiledFn, InlineArray, check_unobserved_error};

fn two_outputs() -> CompiledFn {
    CompiledFn::new(4, true, |inputs| {
        let x = &inputs[0];
        vec![x.add_scalar(1.0), x.mul_scalar(2.0)]
    })
    .expect("compile")
}

#[test]
fn a_call_returns_only_the_outputs_the_closure_made() {
    let mut f = two_outputs();
    let x = InlineArray::from_f32_slice(&[2.0, 3.0], &[2]);
    let mut outs = f.call(&[&x]).expect("call");
    assert_eq!(outs.len(), 2);
    assert_eq!(outs[0].to_f32_vec(2).unwrap(), vec![3.0, 4.0]);
    assert_eq!(outs[1].to_f32_vec(2).unwrap(), vec![4.0, 6.0]);
    check_unobserved_error().expect("bridge error");
}

/// `call` handed the C++ side `n_outputs_max` slots already holding live
/// scalar arrays, and the C++ side placement-new'd the results over them
/// without destroying what was there: each call leaked `n_outputs_max`
/// arrays. In a decode loop that is memory that never comes back.
#[test]
fn calls_do_not_leak_arrays() {
    let mut f = two_outputs();
    let x = InlineArray::from_f32_slice(&[2.0, 3.0], &[2]);
    for _ in 0..50 {
        f.call(&[&x]).expect("warm up")[0].eval();
    }
    clear_cache();
    let before = get_active_memory();
    for _ in 0..5000 {
        let outs = f.call(&[&x]).expect("call");
        outs[0].eval();
        outs[1].eval();
    }
    clear_cache();
    let grown = get_active_memory().saturating_sub(before);
    // A leaked scalar is at least 16 bytes of the allocator's smallest
    // block; 5000 calls x 4 slots would be over 300 KB.
    assert!(grown < 16 * 1024, "active memory grew by {grown} bytes");
    check_unobserved_error().expect("bridge error");
}

/// The closure returning more arrays than `n_outputs_max` wrote past the end
/// of the output buffer.
#[test]
fn a_closure_with_too_many_outputs_is_an_error() {
    let mut f = CompiledFn::new(1, true, |inputs| {
        vec![inputs[0].add_scalar(1.0), inputs[0].add_scalar(2.0)]
    })
    .expect("compile");
    let x = InlineArray::from_f32_slice(&[2.0, 3.0], &[2]);
    let err = pmetal_bridge::expect_bridge_errors(|| f.call(&[&x]))
        .expect_err("two outputs against a capacity of one");
    assert!(err.to_string().contains("n_outputs_max"), "{err}");
    pmetal_bridge::clear_last_error();
}

/// A panic in the closure MLX is tracing unwound through the C++ frames of the
/// `extern "C"` trampoline, which aborts the process. It now reaches the
/// caller of `call` as the same panic, and the handle still works.
#[test]
fn a_panic_in_the_closure_reaches_the_caller() {
    use std::cell::Cell;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::rc::Rc;

    let explode = Rc::new(Cell::new(true));
    let flag = Rc::clone(&explode);
    let mut f = CompiledFn::new(1, false, move |inputs| {
        if flag.get() {
            panic!("closure exploded");
        }
        vec![inputs[0].add_scalar(1.0)]
    })
    .expect("compile");
    let x = InlineArray::from_f32_slice(&[2.0, 3.0], &[2]);

    let payload = catch_unwind(AssertUnwindSafe(|| f.call(&[&x])))
        .expect_err("the closure's panic must come out of call");
    assert_eq!(
        payload.downcast_ref::<&str>().copied(),
        Some("closure exploded")
    );
    check_unobserved_error().expect("the abandoned call leaves no error behind");

    // Nothing about the handle is poisoned: the next call traces again.
    explode.set(false);
    let mut outs = f.call(&[&x]).expect("call after the panic");
    assert_eq!(outs[0].to_f32_vec(2).unwrap(), vec![3.0, 4.0]);
    check_unobserved_error().expect("bridge error");
}
