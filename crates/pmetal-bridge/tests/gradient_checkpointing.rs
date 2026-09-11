//! Gradient checkpointing must not change the gradients.
//!
//! `mlx::core::checkpoint` discards a function's intermediate activations and
//! recomputes them during the backward pass.  That is only a memory/compute
//! trade if the recomputed backward produces exactly what the retained one
//! would have.  These tests hold it to that.
//!
//! The semantics being pinned down here are the ones that make the feature
//! usable at all: gradients reach the arrays passed as explicit inputs, and
//! **only** those.  Anything the closure captures is a constant to the tape.

use pmetal_bridge::InlineArray;
use pmetal_bridge::inline_array::{checkpoint_apply, value_and_grad};

/// Two-layer MLP: `relu(x @ w1) @ w2`, summed to a scalar.
fn mlp(x: &InlineArray, w1: &InlineArray, w2: &InlineArray) -> InlineArray {
    x.matmul(w1).relu().matmul(w2)
}

fn sample_inputs() -> (InlineArray, InlineArray, InlineArray) {
    let x = InlineArray::from_slice(&[0.5f32, -1.5, 2.0, 0.25, -0.75, 1.0], &[2, 3]);
    let w1 = InlineArray::from_slice(
        &[
            0.1f32, -0.2, 0.3, 0.4, 0.5, -0.6, 0.7, -0.8, 0.9, 1.0, -1.1, 1.2,
        ],
        &[3, 4],
    );
    let w2 = InlineArray::from_slice(&[1.0f32, -0.5, 0.25, 2.0], &[4, 1]);
    (x, w1, w2)
}

fn max_abs_diff(a: &InlineArray, b: &InlineArray) -> f32 {
    a.subtract(b).abs().max(None).item_f32()
}

#[test]
fn checkpointing_does_not_change_the_gradients() {
    let (x, w1, w2) = sample_inputs();

    // Reference: the whole forward on the tape, activations retained.
    let (plain_loss, plain_grads) = value_and_grad(
        |a| mlp(&a[2], &a[0], &a[1]).sum_all(),
        &[w1.clone(), w2.clone()],
        std::slice::from_ref(&x),
    );

    // Same forward, run under checkpoint().  The weights have to be handed in
    // as explicit inputs: a captured array is a constant to `custom_vjp` and
    // would silently come back with a zero gradient.
    let (ckpt_loss, ckpt_grads) = value_and_grad(
        |a| {
            let out = checkpoint_apply(&[a[0].clone(), a[1].clone(), a[2].clone()], |inner| {
                vec![mlp(&inner[2], &inner[0], &inner[1])]
            });
            out[0].sum_all()
        },
        &[w1.clone(), w2.clone()],
        std::slice::from_ref(&x),
    );

    assert!(
        (plain_loss.item_f32() - ckpt_loss.item_f32()).abs() < 1e-6,
        "checkpointed loss {} differs from plain loss {}",
        ckpt_loss.item_f32(),
        plain_loss.item_f32()
    );

    for (i, (plain, ckpt)) in plain_grads.iter().zip(ckpt_grads.iter()).enumerate() {
        assert_eq!(
            plain.shape(),
            ckpt.shape(),
            "gradient {i} changed shape under checkpointing"
        );
        let diff = max_abs_diff(plain, ckpt);
        assert!(
            diff < 1e-6,
            "gradient {i} differs by {diff} under checkpointing"
        );
    }
}

#[test]
fn a_captured_array_receives_no_gradient() {
    // This is the failure mode the safe wrapper exists to prevent, pinned here
    // so nobody "simplifies" the parameter plumbing away later.  `w1` is closed
    // over rather than passed in, so `custom_vjp` treats it as a constant and
    // its gradient comes back zero even though the loss plainly depends on it.
    let (x, w1, w2) = sample_inputs();
    let captured = w1.clone();

    let (_, grads) = value_and_grad(
        |a| {
            let captured = captured.clone();
            let out = checkpoint_apply(&[a[2].clone(), a[1].clone()], move |inner| {
                vec![mlp(&inner[0], &captured, &inner[1])]
            });
            out[0].sum_all()
        },
        &[w1.clone(), w2.clone()],
        std::slice::from_ref(&x),
    );

    let w1_grad_magnitude = grads[0].abs().max(None).item_f32();
    assert_eq!(
        w1_grad_magnitude, 0.0,
        "a captured array unexpectedly received a gradient; if MLX changed this, \
         the parameter-threading in `checkpointed` can be simplified"
    );

    // The array that *was* passed in still gets a real gradient.
    let w2_grad_magnitude = grads[1].abs().max(None).item_f32();
    assert!(
        w2_grad_magnitude > 0.0,
        "an explicitly passed array received no gradient"
    );
}

#[test]
fn checkpointing_survives_nesting() {
    // Blocks of layers checkpoint independently in a real trunk, and the outer
    // graph must still differentiate through the boundary between them.
    let (x, w1, w2) = sample_inputs();

    let (plain_loss, plain_grads) = value_and_grad(
        |a| {
            let h = a[2].matmul(&a[0]).relu();
            h.matmul(&a[1]).sum_all()
        },
        &[w1.clone(), w2.clone()],
        std::slice::from_ref(&x),
    );

    let (ckpt_loss, ckpt_grads) = value_and_grad(
        |a| {
            let first = checkpoint_apply(&[a[2].clone(), a[0].clone()], |inner| {
                vec![inner[0].matmul(&inner[1]).relu()]
            });
            let second = checkpoint_apply(&[first[0].clone(), a[1].clone()], |inner| {
                vec![inner[0].matmul(&inner[1])]
            });
            second[0].sum_all()
        },
        &[w1.clone(), w2.clone()],
        std::slice::from_ref(&x),
    );

    assert!(
        (plain_loss.item_f32() - ckpt_loss.item_f32()).abs() < 1e-6,
        "nested checkpointing changed the loss"
    );
    for (i, (plain, ckpt)) in plain_grads.iter().zip(ckpt_grads.iter()).enumerate() {
        let diff = max_abs_diff(plain, ckpt);
        assert!(
            diff < 1e-6,
            "gradient {i} differs by {diff} across a checkpoint boundary"
        );
    }
}
