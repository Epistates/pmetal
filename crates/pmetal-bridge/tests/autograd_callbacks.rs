//! The closures `value_and_grad` and `checkpoint_apply` hand to MLX.

use std::panic::{AssertUnwindSafe, catch_unwind};

use pmetal_bridge::inline_array::{
    checkpoint_apply, clear_cache, get_active_memory, value_and_grad,
};
use pmetal_bridge::{InlineArray, check_unobserved_error};

/// A panic in the loss closure unwound through MLX's C++ frames, inside an
/// `extern "C"` trampoline, which aborts the process: a bug in a loss killed
/// the training run with no chance to save a checkpoint. It now comes out of
/// `value_and_grad` as the same panic, and the next step runs normally.
#[test]
fn a_panic_in_the_loss_reaches_the_caller() {
    let w = InlineArray::from_f32_slice(&[1.0, 2.0], &[2]);

    let payload = catch_unwind(AssertUnwindSafe(|| {
        value_and_grad(
            |_: &[InlineArray]| panic!("loss exploded"),
            std::slice::from_ref(&w),
            &[],
        )
    }))
    .expect_err("the loss's panic must come out of value_and_grad");
    assert_eq!(
        payload.downcast_ref::<&str>().copied(),
        Some("loss exploded")
    );
    check_unobserved_error().expect("the abandoned call leaves no error behind");

    let (loss, mut grads) = value_and_grad(
        |p: &[InlineArray]| p[0].multiply(&p[0]).sum_all(),
        std::slice::from_ref(&w),
        &[],
    );
    assert_eq!(loss.item_f32(), 5.0);
    assert_eq!(grads[0].to_f32_vec(2).unwrap(), vec![2.0, 4.0]);
    check_unobserved_error().expect("bridge error");
}

#[test]
fn a_panic_in_a_checkpointed_block_reaches_the_caller() {
    let x = InlineArray::from_f32_slice(&[1.0, 2.0], &[2]);
    let payload = catch_unwind(AssertUnwindSafe(|| {
        checkpoint_apply(std::slice::from_ref(&x), |_| panic!("block exploded"))
    }))
    .expect_err("the block's panic must come out of checkpoint_apply");
    assert_eq!(
        payload.downcast_ref::<&str>().copied(),
        Some("block exploded")
    );
    check_unobserved_error().expect("the abandoned call leaves no error behind");
}

/// `value_and_grad` handed C++ a loss slot and one slot per parameter that
/// already held live scalar arrays, and C++ placement-new'd the results over
/// them: every step leaked one array per parameter plus one.
#[test]
fn a_step_does_not_leak_arrays() {
    const PARAMS: usize = 32;
    let params: Vec<InlineArray> = (0..PARAMS)
        .map(|i| InlineArray::from_f32_slice(&[i as f32, 1.0], &[2]))
        .collect();

    let step = || {
        let (loss, grads) = value_and_grad(
            |p: &[InlineArray]| {
                let mut total = p[0].multiply(&p[0]).sum_all();
                for w in &p[1..] {
                    total = total.add(&w.multiply(w).sum_all());
                }
                total
            },
            &params,
            &[],
        );
        loss.eval();
        for g in &grads {
            g.eval();
        }
    };

    for _ in 0..20 {
        step();
    }
    clear_cache();
    let before = get_active_memory();
    for _ in 0..200 {
        step();
    }
    clear_cache();
    let grown = get_active_memory().saturating_sub(before);
    // 200 steps x 33 leaked scalars would be 26 KB of 4-byte arrays.
    assert!(grown < 4 * 1024, "active memory grew by {grown} bytes");
    check_unobserved_error().expect("bridge error");
}
